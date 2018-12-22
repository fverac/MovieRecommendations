from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import GridSearchCV
import pandas as pd




df = pd.read_csv("../trainset.csv")
reader = Reader(rating_scale=(1, 5)) #for pandas only
data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

#gridsearchcv on training set to find best params

param_grid = {'reg_all': [0.02, 0.1, 1, 10, 100, 10000]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['mae'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['mae'])
