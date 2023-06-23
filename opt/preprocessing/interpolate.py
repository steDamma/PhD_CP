import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm

def interpolate_cubic(X, single_instance_defined_by, length=None):
    groups = X.groupby(single_instance_defined_by)
    if length is None:
        length = int(groups.size().mode())
    
    rows, cols = length, X.shape[1]
    ret = []
    
    for _, g in tqdm(groups):
        comodo = np.zeros((rows, cols))
        comodo[:rows, :len(single_instance_defined_by)] = np.tile(g.iloc[0, :len(single_instance_defined_by)].values, (rows, 1))

        x = g.index
        x_new = np.linspace(x.min(), x.max(), length)
        
        y = g.iloc[:, len(single_instance_defined_by):].values
        f = CubicSpline(x, y, axis=0)
        
        y_new = f(x_new)
        comodo[:, len(single_instance_defined_by):] = y_new
        
        ret.append(comodo)

    ret = pd.DataFrame(np.vstack(ret), columns=X.columns)
    ret.sort_index(inplace=True)
    ret.reset_index(drop=True, inplace=True)
    return ret, length
