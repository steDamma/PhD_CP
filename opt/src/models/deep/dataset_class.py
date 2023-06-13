import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def leave_one_group_out(data, group_by):
    if group_by not in ['intention', 'ball']:
        raise ValueError('Leave One Out Not Supported')

    loo_groups = data.groupby(['n_groups', 'ball', group_by])
    return list(loo_groups.indices.values())

class MyDataset(Dataset):
    def __init__(self, df, target_column, group_columns, transform=None):
        self.df = df
        self.transform = transform
        self.features, self.labels = self._extract_features_labels(target_column, group_columns)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx])
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label

    def _extract_features_labels(self, target_column, group_columns):
        groups = self.df.groupby(group_columns)
        features, labels = [], []

        for _, group in groups:
            features_ = group.iloc[:, len(group_columns):].values
            label = group[target_column].iloc[0]

            features.append(features_)
            labels.append(label)

        return features, labels
    
    def get_loader(self, config, indices):
        subset = Subset(self.df, indices)

        loader = DataLoader(subset, 
                            batch_size=config.classifier.hyperparameters.batch_size,
                            collate_fn=CustomCollate(),
                            pin_memory=torch.cuda.is_available()
                )
        
        return loader
    '''
    def get_loaders(self, config):
        all_groups = self.df.groupby([self.df.n_groups, self.df.launcher, self.df.receiver, self.df.ball, self.df.intention, self.df.n_launch])
        all_keys = pd.DataFrame(all_groups.groups.keys(), columns=['n_groups', 'launcher', 'receiver', 'ball', 'intention', 'n_launch'])
        loo = config.project.loo_criterion
        loo_idxs = leave_one_group_out(all_keys, group_by=loo)

        test_loaders = []
        train_loaders = []

        for i in range(len(loo_idxs)):
            train_indices = np.setdiff1d(np.arange(len(self.df)), loo_idxs[i])
            test_indices = loo_idxs[i]

            train_dataset = Subset(self, train_indices)
            test_dataset = Subset(self, test_indices)

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.classifier.hyperparameters.batch_size,
                collate_fn=CustomCollate(),
                pin_memory=torch.cuda.is_available()
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.classifier.hyperparameters.batch_size,
                collate_fn=CustomCollate(),
                pin_memory=torch.cuda.is_available()
            )

            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
        
        return train_loaders, test_loaders
    '''
class CustomCollate:
    def __call__(self, data):
        inputs = [torch.tensor(d[0]) for d in data]
        labels = [torch.tensor(d[1]) for d in data]

        inputs = pad_sequence(inputs, batch_first=True)
        labels = torch.tensor(labels)
        return inputs, labels
