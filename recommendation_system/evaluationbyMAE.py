from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import GridSearchCV
import pandas as pd





reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5), skip_lines = 1)

fold_files = [("../trainset.csv","../testset.csv")];
data = Dataset.load_from_folds(fold_files, reader=reader)
algo = SVD(reg_all = 0.02)

pkf = PredefinedKFold()
for trainset, testset in pkf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Mean Absolute Error
    accuracy.mae(predictions, verbose=True)