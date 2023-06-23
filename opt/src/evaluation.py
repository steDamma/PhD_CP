import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score as ACC, balanced_accuracy_score as BACC, roc_auc_score as ROC, f1_score 

def evaluate(y_true, y_pred, y_pred_proba, num_trials):
    all_accuracies = []
    all_balanced_accuracies = []
    all_rocs = {}
    all_f1_scores = {}
    metrics = {}

    for _ in range(num_trials):
        # Bootstrap sampling
        y_true_resampled, y_pred_resampled, y_pred_proba_resampled = resample(y_true, y_pred, y_pred_proba, replace=True)

        accuracies = []
        balanced_accuracies = []
        rocs = {}
        f1_scores = {}

        accuracy = ACC(y_true_resampled, y_pred_resampled)
        balanced_accuracy = BACC(y_true_resampled, y_pred_resampled)

        for cl in np.unique(y_true):
            roc = ROC(y_true_resampled == cl, y_pred_proba_resampled[:, int(cl)], multi_class='ovr')
            rocs.setdefault(cl, []).append(roc)

            f1 = f1_score(y_true_resampled == cl, y_pred_resampled, average='binary')
            f1_scores.setdefault(cl, []).append(f1)

        accuracies.append(accuracy)
        balanced_accuracies.append(balanced_accuracy)

        all_accuracies.extend(accuracies)
        all_balanced_accuracies.extend(balanced_accuracies)

        for key in rocs.keys():
            if key in all_rocs:
                all_rocs[key].extend(rocs[key])
            else:
                all_rocs[key] = rocs[key]

            if key in all_f1_scores:
                all_f1_scores[key].extend(f1_scores[key])
            else:
                all_f1_scores[key] = f1_scores[key]

    metrics['ACC_mean'] = np.mean(all_accuracies)
    metrics['ACC_std'] = np.std(all_accuracies)
    metrics['BACC_mean'] = np.mean(all_balanced_accuracies)
    metrics['BACC_std'] = np.std(all_balanced_accuracies)

    for key in all_rocs.keys():
        metrics[f'ROC_class{key}_mean'] = np.mean(all_rocs[key])
        metrics[f'ROC_class{key}_std'] = np.std(all_rocs[key])

        metrics[f'F1_score_class{key}_mean'] = np.mean(all_f1_scores[key])
        metrics[f'F1_score_class{key}_std'] = np.std(all_f1_scores[key])

    return metrics
