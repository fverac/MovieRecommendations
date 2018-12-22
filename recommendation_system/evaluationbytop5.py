from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy



from collections import defaultdict
import pandas as pd


#This file gives the top 5 movie recommendations for each user and evaluates these recommendations by
#aggregating and averaging the recommendations found in the test set.


def get_top_n(predictions, n=10):
	'''Return the top-N recommendation for each user from a set of predictions.

	Args:
		predictions(list of Prediction objects): The list of predictions, as
			returned by the test method of an algorithm.
		n(int): The number of recommendation to output for each user. Default
			is 10.

	Returns:
	A dict where keys are user (raw) ids and values are lists of tuples:
		[(raw item id, rating estimation), ...] of size n.
	'''

	# First map the predictions to each user.
	top_n = defaultdict(list)
	for uid, iid, true_r, est, _ in predictions:
		top_n[uid].append((iid, est))

	# Then sort the predictions for each user and retrieve the k highest ones.
	for uid, user_ratings in top_n.items():
		user_ratings.sort(key=lambda x: x[1], reverse=True)
		top_n[uid] = user_ratings[:n]

	return top_n



testdf = pd.read_csv("../testset.csv")
df = pd.read_csv("../trainset.csv")

reader = Reader(rating_scale=(1, 5)) #for pandas only
data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)


algo = SVD(reg_all = 0.02)

trainset = data.build_full_trainset()
algo.fit(trainset)



# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=5)



#for every user's movie recommendation, check if the user movie pair is in the test set.
#if in the test set, add the rating to the sum. if the recommendation is not in the test set, skip.
#finally take the average and print
sum = 0
totusers = 943
totrecs = 0

for uid, user_ratings in top_n.items():
	userid = uid
	itemids = [iid for (iid, _) in user_ratings]

	for itemid in itemids:

		searchintest = testdf.loc[testdf.user == float(userid)].loc[testdf.item == float(itemid)]

		score = 2;
		print(score)
		if (not searchintest.empty):
			score = searchintest.rating.values[0];
			totrecs = totrecs+1
			sum = sum + score

		

print(sum/totrecs)