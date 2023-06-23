import numpy as np
import pandas as pd
from scipy.stats import mode

import torch
from torch.utils.data import Dataset, Subset, DataLoader

class MyDataset(Dataset):
    def __init__(self, df, target_column, single_istance_defined_by, transform=None):
        self.df = df
        self.transform = transform
        self.single_istance_defined_by = list(single_istance_defined_by)  
        self.target_column = target_column

        self.groups = self.df.groupby(self.single_istance_defined_by)
        self.groups_keys = pd.DataFrame(self.groups.groups.keys(), columns=self.single_istance_defined_by)

        self.features, self.labels = self._extract_features_labels()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx])
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label

    def _extract_features_labels(self):
        lengths, features, labels = [], [], []

        for _, group in self.groups:
            features_ = group.iloc[:, len(self.single_istance_defined_by):].values
            label = group[self.target_column].iloc[0]

            lengths.append(len(features_))
            features.append(features_)
            labels.append(label)

        self.target_length = mode(lengths).mode[0]
        features = np.asarray(features)
        labels = np.asarray(labels)

        return features, labels

    def get_loader(self, config, indices):
        subset = Subset(self, indices)

        loader = DataLoader(
            subset,
            batch_size=config.classifier.hyperparameters.batch_size,
            collate_fn=self.custom_collate,
            pin_memory=torch.cuda.is_available()
        )

        return loader

    
    def custom_collate(self, data):
        inputs = [d[0].detach().clone() for d in data]
        labels = [torch.tensor(d[1]) for d in data]

        inputs = torch.stack(inputs).float()
        labels = torch.stack(labels).long()
        return inputs, labels
    
    def leave_one_group_out(self, loo, filter_idx=None, n_samples=-1):
        filtered_groups = self.df.groupby(self.single_istance_defined_by, as_index=False)
        all_idxs = np.arange(len(self))

        filtered_keys = pd.DataFrame(filtered_groups.groups.keys(), columns=self.single_istance_defined_by)

        loo_groups_idx = list(filtered_keys.groupby(loo).indices.values())
        if filter_idx is not None:
            loo_groups_idx = [np.setdiff1d(filter_idx, group_idx) for group_idx in loo_groups_idx if len(np.setdiff1d(filter_idx, group_idx)) > 0]
            all_idxs = filter_idx

        splits = []
        mc = np.random.choice(len(loo_groups_idx), size=n_samples, replace=False) if n_samples != -1 else np.arange(len(loo_groups_idx))

        for i in range(len(loo_groups_idx)):
            train_idx = np.setdiff1d(all_idxs, loo_groups_idx[i]).tolist()
            if i in mc and len(train_idx) > 0:
                test_idx = loo_groups_idx[i].tolist() 

                splits.append([train_idx, test_idx])

        return splits


