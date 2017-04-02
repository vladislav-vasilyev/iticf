# iticf
Implementation of the recommendation system for the movies dataset. It implements item to item collaborative filtering algorithm, since it gives predictions with higher accuracy then user-user collaborative filtering algorithm.
As input file it takes information from u1.base which contains in http://files.grouplens.org/datasets/movielens/ml-100k.zip archive.
The source page: https://grouplens.org/datasets/movielens/100k/


# Breif Description of the Algorithm
Firstly, the data is unfolded in the next way: columns correspond to the users, rows correspond to the movies, and values at the intersection represent the rating of the movie rated by the user.
Then algorithm normalizes the ratings for each movie by subtracting the mean movie rating from each of its ratings. The data normalization by mean movie rating gives smaller RMSE than in case with normalization by mean user rating.
Then we calculate the similarity using the cosine distance. Empirically I have discovered that the cosine distance gives the same results as Pearson distance for the provided data. But since the cosine distance is computationally simpler it was used for the end solution.
The obtained table has equal amount of columns and rows, since each value of this table shows the similarity factor of the movie’s ratings. That is why the diagonal values are all equal to 1.
With the use of the calculated similarity table we provide recommendations.
