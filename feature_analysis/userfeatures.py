from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression



#code to try and predict user gender given the learned user matrix

df = pd.read_csv("../trainset.csv")
gender = pd.read_csv("../gender.csv")

reader = Reader(rating_scale=(1, 5)) #for pandas only
data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

algo = SVD(reg_all = 0.02)
trainset = data.build_full_trainset()
algo.fit(trainset)


usermatrix = algo.pu
y= gender.values
clforig = LogisticRegression()
parameters = {'C':[0.01, 0.1, 1, 10, 100, 1000]}
clf = GridSearchCV(clforig, parameters, cv=5) 
clf.fit(usermatrix,y)
results = pd.DataFrame.from_dict(clf.cv_results_)
print(results.transpose())


#best mean accuracy = 0.710498
#best C = 0.01
