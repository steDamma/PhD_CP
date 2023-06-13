from tqdm import tqdm
from omegaconf import OmegaConf

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB

from evaluation import evaluate
from model_selection import leave_1_out, monte_carlo_leave_1_out

class ShallowClassifier:
    SUPPORTED_CLASSIFIERS = {
        'rf': RandomForestClassifier(),
        'svm_lin': SVC(probability=True),
        'svm_rbf': SVC(probability=True),
        'xgboost': xgb.XGBClassifier(use_label_encoder=False, objective="binary:logistic", eval_metric="logloss"),
        'naive_bayes': GaussianNB()
    }
    
    def __init__(self, config):
        self.name = config.classifier.name
        self.param_grid = OmegaConf.to_container(config.classifier.hyperparameters)
        self.info = config
        self.classifier = self.get_classifier()

    def get_classifier(self):
        if self.name not in self.SUPPORTED_CLASSIFIERS:
            raise ValueError(f"Classifier not supported: {self.name}")
        return self.SUPPORTED_CLASSIFIERS[self.name]

    def run_classifier(self, X):
        scores = {}
        loo = list(leave_1_out(X, self.info.project.loo_criterion, self.info.project.target.aim))
        
        outer_bar = tqdm(total=len(loo), desc="Test Sets Predicted")
        inner_bar = tqdm(total=self.info.num_trials, desc="Validation Trials Performed", leave=False)

        for it, (lr_idx, ts_idx) in enumerate(loo):
            learning_set = X.loc[lr_idx].reset_index(drop=True)
            test = X.loc[ts_idx]

            y_trues, y_preds, y_pred_probas = [], [], []
            
            for _ in range(self.info.num_trials):
                inner_bar.update(1)
                loo_val = monte_carlo_leave_1_out(learning_set, self.info.project.loo_criterion, self.info.project.target.aim, n_samples=self.info.val_groups) 
                clf = GridSearchCV(self.classifier, param_grid=self.param_grid, cv=loo_val, n_jobs=self.info.n_jobs, scoring=self.info.metric_to_optimize)

                X_test = test.drop(self.info.project.target.aim, axis=1).values
                y_test = test[self.info.project.target.aim].values

                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)

                y_trues.extend(y_test.tolist())
                y_preds.extend(y_pred.tolist())
                y_pred_probas.extend(y_pred_proba.tolist())

            outer_bar.update(1)
            inner_bar.reset()
            
            y_trues = np.array(y_trues)
            y_preds = np.array(y_preds)
            y_pred_probas = np.array(y_pred_probas)
            scores[f'{it}'] = evaluate(y_trues, y_preds, y_pred_probas)
        
        outer_bar.close()
        inner_bar.close()
        
        return scores
