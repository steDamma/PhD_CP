import os
import numpy as np
from itertools import permutations,product
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from models.deep.lstm_class import LSTM
from models.deep.tcn_class import TCN
from models.deep.dataset_class import MyDataset

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def get_classifier(self, **hypers):
        if self.name not in self.SUPPORTED_CLASSIFIERS:
            raise ValueError(f"Classifier not supported: {self.name}")
        
        clf = TCN(**hypers) if self.name == 'tcn' else LSTM(**hypers)
        
        return clf.to(self.device)
    
    def evaluate_model(self, dataloader):
        criterion = nn.CrossEntropyLoss()        
        loss_sum = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        y_pred_probas = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                probabilities = nn.functional.softmax(outputs, dim=1)
                
                loss_sum += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predicted.cpu().tolist())
                y_pred_probas.extend(probabilities.cpu().tolist())

        loss_avg = loss_sum / total
        accuracy = correct / total
        
        return loss_avg, accuracy, y_true, y_pred, y_pred_probas


    def train_model(self,
                    train_loader, 
                    val_loader, 
                    test_loader
                    ):
        
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = Adam(parameters, 
                         lr=self.info.classifier.hyperparameters.lr,
                         weight_decay=self.info.classifier.hyperparameters.weight_decay)

        scheduler = OneCycleLR(optimizer, 
                               max_lr=self.info.classifier.hyperparameters.lr,
                               epochs=self.info.classifier.hyperparameters.epochs,
                               steps_per_epoch=len(train_loader))
        
        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        best_acc = 0.0
        project_folder = self.info.project.name
        patience = 100
        trials = 0
        y_preds, y_trues, y_probas = [], [], []
        
        if self.info.classifier.verbose:
            print('\n\n')
        
        for epoch in range(self.info.classifier.hyperparameters.epochs):
            sum_loss = 0.0
            total = 0
            correct = 0

            self.model.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device) 
                labels = labels.to(self.device)

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
            val_loss, val_acc, _, _, _ = self.evaluate_model(val_loader)
            test_loss, test_acc, labels_predicted, true_labels_, probas = self.evaluate_model(test_loader)

            if self.info.classifier.verbose and epoch % 10 == 0:
                print(f'Epoch {epoch} -> Train Loss: {train_loss:.3f}, Train ACC: {train_acc:.2f}, Val Loss:{val_loss:.3f}, Val ACC:{val_acc:.2f}, Test Loss: {test_loss:.3f}, Test ACC: {test_acc:.2f}')
                        
            if val_loss < best_loss:
                trials = 0
                best_loss = val_loss
                best_acc = val_acc
                y_preds = labels_predicted
                y_trues = true_labels_
                y_probas = probas

                if self.info.classifier.store_model:
                    if not os.path.exists(project_folder):
                        os.makedirs(project_folder)                
                    file_path = os.path.join(project_folder, f'model_clf_{self.info.project.loo_criterion}_target_{self.info.project.target.aim}.pth')
                    torch.save(self.model.state_dict(), file_path)

            else:
                trials += 1
                if trials >= patience:
                    break
        
        torch.cuda.empty_cache()
        return best_acc, y_preds, y_trues, y_probas

    def run_classifier(self, df):
        dataset = MyDataset(df=df, 
                            target_column=self.info.project.target.aim, 
                            single_istance_defined_by=self.info.project.single_istance_defined_by, 
                            transform=None)

        fixed_values = {
            'input_dim': dataset.target_length,
            'output_size': len(np.unique(dataset.df[self.info.project.target.aim].values)),
            'num_features': dataset.df.shape[1] - len(self.info.project.single_istance_defined_by),
            'device': self.device
        }

        keys = self.param_grid.keys()
        values = self.param_grid.values()

        if self.name == 'tcn':
            kernel_sizes = OmegaConf.to_container(self.info.classifier.hyperparameters.kernel_sizes)
            kernel_combinations = list(permutations(kernel_sizes, fixed_values['num_features']))

        combinations = []

        for hypers in product(*[v if isinstance(v, list) else [v] for v in values]):
            hypers_dict = dict(zip(keys, hypers))

            if self.name == 'tcn':
                for kernel_comb in kernel_combinations:
                    hypers_dict['kernel_sizes'] = list(kernel_comb)
                    hypers_dict.update(fixed_values)
                    combinations.append(hypers_dict.copy())
            else:
                hypers_dict.update(fixed_values)
                combinations.append(hypers_dict.copy())
                
        len_comb = len(combinations)
        loo = dataset.leave_one_group_out(self.info.project.loo_criterion)

        scores = {}
        
        for itest, (lr_idx, ts_idx) in tqdm(enumerate(loo), desc="Test Sets Predicted"):
            test_loader = dataset.get_loader(self.info, ts_idx)

            y_trues, y_preds, y_pred_probas = [[]] * len_comb, [[]] * len_comb, [[]] * len_comb
            val_scores = np.zeros((len_comb))

            for ih, hypers in enumerate(combinations):
                hypers.update(fixed_values)
                
                if self.name == 'tcn':
                    model_dict = TCN().__dict__
                else:
                    model_dict = LSTM().__dict__

                for key in list(hypers.keys()):
                    if key not in model_dict:
                        hypers.pop(key)

                loo_val = dataset.leave_one_group_out(self.info.project.loo_criterion, lr_idx, n_samples=self.info.val_groups)

                for _, (train_idx, val_idx) in tqdm(enumerate(loo_val), desc="Validation Trials Performed", leave=False):
                    self.model = self.get_classifier(**hypers)
                    train_loader = dataset.get_loader(self.info, train_idx)
                    val_loader = dataset.get_loader(self.info, val_idx)

                    val_acc, y_pred, y_true, y_proba = self.train_model(train_loader, 
                                                                        val_loader, 
                                                                        test_loader)

                    val_scores[ih] += val_acc

                    y_trues[ih].extend(y_true) 
                    y_preds[ih].extend(y_pred)
                    y_pred_probas[ih].extend(y_proba)
            
            val_accs = np.mean(val_scores, axis=0)
            best_hyper = np.argmax(val_accs)

            y_trues = np.array(y_trues[best_hyper])
            y_preds = np.array(y_preds[best_hyper])
            y_pred_probas = np.array(y_pred_probas[best_hyper])

            scores[f'{itest}'] = evaluate(y_trues, y_preds, y_pred_probas, self.info.num_trials)

        return scores