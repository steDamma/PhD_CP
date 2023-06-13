import os
import numpy as np
import random
import itertools
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from deep.lstm_class import LSTM
from deep.tcn_class import TCN
from deep.dataset_class import MyDataset

from evaluation import evaluate
from model_selection import leave_1_out, monte_carlo_leave_1_out

class DeepClassifier:
    SUPPORTED_CLASSIFIERS = {
        'lstm': LSTM(),
        'lstm_bidirectional': LSTM(),
        'tcn': TCN()
    }

    def __init__(self, config):
        self.name = config.classifier.name
        self.param_grid = OmegaConf.to_container(config.classifier.hyperparameters)
        self.info = config
        self.model = self.get_classifier()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
    
    def get_classifier(self):
        if self.name not in self.SUPPORTED_CLASSIFIERS:
            raise ValueError(f"Classifier not supported: {self.name}")
        return self.SUPPORTED_CLASSIFIERS[self.name]
    
    def evaluate_model(self, dataloader):
        criterion = nn.CrossEntropyLoss()        
        loss_sum = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        y_pred_probas = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                probabilities = nn.functional.softmax(outputs, dim=1)
                
                loss_sum += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())
                y_pred_probas.extend(probabilities.tolist())

        loss_avg = loss_sum / total
        accuracy = correct / total
        
        return loss_avg, accuracy, y_true, y_pred, y_pred_probas


    def train_model(self,
                    train_loader, 
                    val_loader, 
                    test_loader, 
                    config,
                    store_model:int = 0,
                    verbose:int = 0):
        
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = Adam(parameters, 
                         lr=config.classifier.hyperparameters.lr,
                         weight_decay=config.classifier.hyperparameters.weight_decay)
        
        scheduler = OneCycleLR(optimizer, 
                               max_lr=config.classifier.hyperparameters.lr,
                               epochs=config.classifier.hyperparameters.epochs,
                               steps_per_epoch=len(train_loader))
        
        criterion = nn.CrossEntropyLoss()
        
        best_loss = float('inf')
        best_acc = 0.0
        project_folder = config.project.name
        patience = 100
        trials = 0
        y_preds, y_trues, y_probas = [], [], []
        
        self.model.train()
        for epoch in range(config.classifier.hyperparameters.epochs):
            sum_loss = 0.0
            total = 0
            correct = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                sum_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            
            train_loss = sum_loss/total
            train_acc = correct/total
            val_loss, val_acc, _, _, _ = self.evaluate_model(self.model, val_loader)
            test_loss, test_acc, labels_predicted, true_labels_, probas = self.evaluate_model(self.model, test_loader)

            if verbose:
                print(f'Epoch {epoch} -> Train Loss: {train_loss}, Train ACC: {train_acc}, Val Loss:{val_loss}, Val ACC:{val_acc}, Test Loss: {test_loss}, Test ACC: {test_acc}')
                        
            if val_loss < best_loss:
                trials = 0
                best_loss = val_loss
                best_acc = val_acc
                y_preds = labels_predicted
                y_trues = true_labels_
                y_probas = probas

                if store_model:
                    if not os.path.exists(project_folder):
                        os.makedirs(project_folder)                
                    file_path = os.path.join(project_folder, f'model_clf_{config.project.loo_criterion}_target_{config.project.target.aim}.pth')
                    torch.save(self.model.state_dict(), file_path)

            else:
                trials += 1
                if trials >= patience:
                    break
        
        torch.cuda.empty_cache()
        return best_acc, y_preds.cpu(), y_trues.cpu(), y_probas.cpu()

    def run_classifier(self, df, config):
        dataset = MyDataset(df=df, target_column=config.project.target.aim, group_columns=config.project.group_columns, transform=None)

        keys = self.param_grid.keys()
        values = self.param_grid.values()

        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        len_comb = len(combinations)
        loo = list(leave_1_out(df, self.info.project.loo_criterion, self.info.project.target.aim))

        scores = {}

        outer_bar = tqdm(total=len(loo), desc="Test Sets Predicted")
        inner_bar = tqdm(total=self.info.num_trials, desc="Validation Trials Performed", leave=False)

        for itest, (lr_idx, ts_idx) in enumerate(loo):
            learning_set = df.loc[lr_idx].reset_index(drop=True)
            test_loader = dataset.get_loader(self.info, ts_idx)

            y_trues, y_preds, y_pred_probas = [[]] * len_comb, [[]] * len_comb, [[]] * len_comb
            val_scores = np.zeros((len_comb, self.info.num_trials))

            for ih, hypers in enumerate(combinations):
                self.model(**hypers)

                for itrials in range(self.info.num_trials):
                    inner_bar.update(1)
                    loo_val = list(monte_carlo_leave_1_out(learning_set, self.info.project.loo_criterion, self.info.project.target.aim, n_samples=self.info.val_groups))
            
                    for _, (train_idx, val_idx) in enumerate(loo_val):
                        train_loader = dataset.get_loader(self.info, train_idx)
                        val_loader = dataset.get_loader(self.info, val_idx)

                        val_acc, y_pred, y_true, y_proba = self.train_model(train_loader, 
                                                                    val_loader, 
                                                                    test_loader, 
                                                                    self.info,
                                                                    store_model = self.info.classifier.store_model,
                                                                    verbose = self.info.classifier.verbose)

                        val_scores[ih,itrials] += val_acc

                        y_trues[ih].extend(y_true.tolist()) 
                        y_preds[ih].extend(y_pred.tolist())
                        y_pred_probas[ih].extend(y_proba.tolist())
            
            val_accs = np.mean(val_scores, axis=0)
            best_hyper = np.argmax(val_accs)

            y_trues = np.array(y_trues[best_hyper])
            y_preds = np.array(y_preds[best_hyper])
            y_pred_probas = np.array(y_pred_probas[best_hyper])
            
            outer_bar.update(1)
            inner_bar.reset()

            scores[f'{itest}'] = evaluate(y_trues, y_preds, y_pred_probas)

        outer_bar.close()
        inner_bar.close()

        return scores