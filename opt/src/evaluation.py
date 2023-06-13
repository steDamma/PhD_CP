import numpy as np
from sklearn.metrics import accuracy_score as ACC, balanced_accuracy_score as BACC, roc_auc_score as ROC, f1_score 

def evaluate(y_true, y_pred, y_pred_proba):
    accuracies = []
    balanced_accuracies = []
    rocs = {}
    f1_scores = {}
    metrics = {}
    
    accuracy = ACC(y_true, y_pred)
    balanced_accuracy = BACC(y_true, y_pred)

    for cl in np.unique(y_true):
        roc = ROC(y_true == cl, y_pred_proba[:, int(cl)], multi_class='ovr')
        rocs.setdefault(cl, []).append(roc)

        f1 = f1_score(y_true == cl, y_pred, average='binary')
        f1_scores.setdefault(cl, []).append(f1)

    accuracies.append(accuracy)
    balanced_accuracies.append(balanced_accuracy)

    metrics['ACC_mean'] = np.mean(accuracies)
    metrics['ACC_std'] = np.std(accuracies)
    metrics['BACC_mean'] = np.mean(balanced_accuracies)
    metrics['BACC_std'] = np.std(balanced_accuracies)

    for key in rocs.keys():
        metrics[f'ROC_class{key}_mean'] = np.mean(rocs[key])
        metrics[f'ROC_class{key}_std'] = np.std(rocs[key])

        metrics[f'F1_score_class{key}_mean'] = np.mean(f1_scores[key])
        metrics[f'F1_score_class{key}_std'] = np.std(f1_scores[key])

    return metrics
