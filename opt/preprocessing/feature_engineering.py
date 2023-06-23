import numpy as np
from scipy.stats import kurtosis, skew
from scipy.stats import iqr

def extractFeatures(features):
  raw_features = features.shape[1]
  arr = features.values
  energy = np.sqrt(np.sum(arr ** 2, axis=0))
  fft = np.fft.fft(arr, axis=0)
  amplitude_spectrum = np.abs(fft)
  phase_angle = np.angle(fft)

  frq_info = [
              phase_angle[0, :],  
              np.mean(fft.real, axis=0),
              np.max(fft.real, axis=0),
              np.argmax(fft.real, axis=0),
              np.min(fft.real, axis=0),
              np.argmin(fft.real, axis=0),
              skew(amplitude_spectrum, axis=0, bias=True),
              kurtosis(amplitude_spectrum, axis=0, bias=True),
  ]

  frq_info = np.hstack(frq_info)
  mean = np.mean(arr, axis=0)
  var = np.var(arr, axis=0)
  kurt = kurtosis(arr, axis=0, bias=True)
  skew_ = skew(arr, axis=0, bias=True)
  corr = np.corrcoef(arr, rowvar=False)[np.triu_indices(raw_features,k=1)]
  mad = np.mean(np.abs(arr - mean), axis=0)
  sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
  mi = np.min(arr, axis=0)
  ma = np.max(arr, axis=0)

  return np.hstack([mean,var,kurt,skew_,corr,mad,sem,energy,iqr(arr,axis=0),mi,ma,frq_info])

def get_feature_names(df, metadata):
  df_raw_feat = df.columns
  new_names = list(df_raw_feat[:metadata])
  raw_features = df_raw_feat[metadata:]

  # Define the feature names based on the order they appear in extractFeatures
  new_names += [f"mean_{i}" for i in raw_features]
  new_names += [f"var_{i}" for i in raw_features]
  new_names += [f"kurt_{i}" for i in raw_features]
  new_names += [f"skew_{i}" for i in raw_features]
  new_names += [f"corr_{i_name}_{j_name}" for i, i_name in enumerate(raw_features[:-1]) for j, j_name in enumerate(raw_features[i+1:])]
  new_names += [f"mad_{i}" for i in raw_features]
  new_names += [f"sem_{i}" for i in raw_features]
  new_names += [f"energy_{i}" for i in raw_features]
  new_names += [f"iqr_{i}" for i in raw_features]
  new_names += [f"min_{i}" for i in raw_features]
  new_names += [f"max_{i}" for i in raw_features]
  
  new_names += [f"fft_phase_angle_{i}" for i in raw_features]
  new_names += [f"fft_mean_real{i}" for i in raw_features]
  new_names += [f"fft_max_real{i}" for i in raw_features]
  new_names += [f"fft_argmax_real{i}" for i in raw_features]
  new_names += [f"fft_min_real{i}" for i in raw_features]
  new_names += [f"fft_argmin_real{i}" for i in raw_features]
  new_names += [f"fft_skew_amp_spec{i}" for i in raw_features]
  new_names +=  [f"fft_kurt_amp_spec{i}" for i in raw_features]
  
  return new_names