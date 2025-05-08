import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy/src/main/resources/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy/src/main/resources/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

#print(ratings.head())

movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
#print(movieRatings.head())

starWarsRatings = movieRatings['Star Wars (1977)']
#print(starWarsRatings.head())

similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
similarMovies.sort_values(ascending=False)
df = pd.DataFrame(similarMovies)
#print(df.head(10))


import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
#print(movieStats.head())

popularMovies = movieStats['rating']['size'] >= 100
#print(movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15])

mappedColumnsMoviestat=movieStats[popularMovies]
mappedColumnsMoviestat.columns=[f'{i}|{j}' if j != '' else f'{i}' for i,j in mappedColumnsMoviestat.columns]
df = mappedColumnsMoviestat.join(pd.DataFrame(similarMovies, columns=['similarity']))

#print(df.head())
print(df.sort_values(['similarity'], ascending=False)[:15])