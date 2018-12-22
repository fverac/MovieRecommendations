from surprise import SVD
from surprise import Dataset
from surprise import Reader

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge

import numpy as np


testdf = pd.read_csv("../testset.csv")
df = pd.read_csv("../trainset.csv")
combined = pd.concat([testdf,df])

years = pd.read_csv("../release-year.csv")

reader = Reader(rating_scale=(1, 5)) #for pandas only
data = Dataset.load_from_df(combined[['user', 'item', 'rating']], reader)

algo = SVD(reg_all = 0.02)

trainset = data.build_full_trainset()
algo.fit(trainset)

moviematrix = algo.qi
y = years.values

#KERNEL RIDGE regression for release year
#best mean test MSE: 214.434
#best test MSE for a single split: 116.92
parameters = {"gamma": [1e0, 0.1, 1e-2, 1e-3, 1e-4,1e-6]}
kr = KernelRidge(kernel = 'rbf')
clf = GridSearchCV(kr,parameters, cv = 5, scoring = 'neg_mean_squared_error');

clf.fit(moviematrix,y);
results = pd.DataFrame.from_dict(clf.cv_results_)
print(results.transpose())

#NAIVE movie release year estimation
naive = np.ones(1681)
naive = naive * y.mean()
print( mean_squared_error(naive, y) )
#mse is 203.04368486931486