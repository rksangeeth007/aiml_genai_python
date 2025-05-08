import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy/src/main/resources/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy/src/main/resources/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

ratings.head()

userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()

corrMatrix = userRatings.corr()
corrMatrix.head()

#This is to avoid spurious result by picking only movies where fewer than 100 users rated a given movie pair
corrMatrix = userRatings.corr(method='pearson', min_periods=100)
#print(corrMatrix.head())

myRatings = userRatings.loc[0].dropna()
#myRatings


simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates; May be below code might not work as append is deprecated from Panda 2*
    simCandidates = simCandidates.append(sims)


print ("sorting...")
simCandidates.sort_values(inplace = True, ascending = False)
print (simCandidates.head(10))


simCandidates = simCandidates.groupby(simCandidates.index).sum()
simCandidates.sort_values(inplace = True, ascending = False)
#simCandidates.head(10)

filteredSims = simCandidates.drop(myRatings.index)
#print(filteredSims.head(10))
