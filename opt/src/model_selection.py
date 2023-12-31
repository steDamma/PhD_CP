import numpy as np

def leave_1_out(X,loo, target):
  '''
  - X a pd.DataFrame
  - loo column name

  This function takes a pd.DataFrame as input and returns groups considering all columns up to the one defined in 'loo'.
  '''
  
  target_column_index = X.columns.get_loc(loo)
  columns_to_group = X.columns[:target_column_index + 1].tolist()

  grouped = X.groupby(columns_to_group)
  for i, (name,test) in enumerate(grouped):
    train = X.drop(test.index)
    train_idx = train.index.intersection(X.index).tolist()
    test_idx = test.index.tolist()
    yield train_idx, test_idx


def monte_carlo_leave_1_out(X, loo, target, n_samples):
  target_column_index = X.columns.get_loc(loo)
  columns_to_group = X.columns[:target_column_index + 1].tolist()

  grouped = X.groupby(columns_to_group)
  lun = len(grouped)
  mc = np.random.choice(lun, size=n_samples, replace=False)

  for i, (name,test) in enumerate(grouped):
    if i in mc:
      train = X.drop(test.index)
      yield train.index, test.index




