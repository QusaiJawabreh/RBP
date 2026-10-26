# RBP METHOD
from sklearn import svm 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def RBP(dataframe, *, n_peel, standardize=True, performPCA=False):
  if standardize == True:                               # standardize the data to mean 0 and variance 1 for each feature 
    scaler = StandardScaler()
    dataframe = scaler.fit_transform(dataframe) 

  if performPCA == True:                                # reduce dimension of data such to the least possible PCs that explain at least 99% of variance
    reducer = PCA(n_components=0.99, svd_solver='full')           
    dataframe = reducer.fit_transform(dataframe)

  n = dataframe.shape[0]           # to read the number of observations
  m = n
  p = dataframe.shape[1]           # to read the number of features
  s = 1/p

  copy = dataframe.copy()                                           # create a copy to allow peeling without distorting original data
  ocsvm = svm.OneClassSVM(kernel = 'rbf', gamma = s, nu = 0.01)
  D = np.zeros(n).reshape(n, 1)                                     # Initial D column (will be removed later)
  peel_count = 0

  while m > n_peel:                                                 # keep peeling until no more than n_peel points remain
    peel_count += 1
    w = ocsvm.fit(copy)                                           # Fit OCSVM to current data subset
    support_idx = w.support_                                      # Support vectors in copy index space
    copy = np.delete(copy, tuple(support_idx), axis=0)            # Remove them from copy 
    m = copy.shape[0]                                             # update m to reflect number of observations left
    dec_func_dist = (ocsvm.decision_function(dataframe) * -1)     # Compute distances for all original points
    D = np.concatenate([D, dec_func_dist.reshape(-1,1)], axis=1)  # distance matrix

  D = D[:, 1:]                                             # remove the first column (created before peeling)
  num_peels = D.shape[1]                                   # store the number of peels
  KD = np.median(D, axis=1)                                # compute average distance via median for each observation
  M = np.median(KD)                                        # find the median value for all observations
  upper_mask = KD > M                                      # consider only the upper half of observation scores (double MAD)
  if np.sum(upper_mask) == 0:                              # If there are no observations above the median, fall back to the standard MAD.
    MAD_upper = np.median(np.abs(KD - M))
  else:
    MAD_upper = np.median(np.abs(KD[upper_mask] - M))      # calculate upper MAD, no need for lower since this is a one-way control limit
  if MAD_upper <= np.finfo(float).eps:                     # prevent division by zero
    MAD_upper = np.finfo(float).eps
     
  score = 0.6745 * (KD - M) / MAD_upper                    # calculate modified z-score for each obsevation via double MAD
  Flag = np.where(score < 1.5, 1, -1)                      # assign scores to points, 1 for inliers and -1 for outliers

  return score, Flag, num_peels
  
