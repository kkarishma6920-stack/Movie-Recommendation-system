import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("ml-latest-small/movies.csv")

# Fill empty genres
movies["genres"] = movies["genres"].fillna("")

# Convert genres into numerical vectors
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies["genres"])

# Calculate similarity
similarity = cosine_similarity(count_matrix)

def recommend(movie_title):

    movie_title = movie_title.lower()

    movies["title_lower"] = movies["title"].str.lower()

    if movie_title not in movies["title_lower"].values:
        return ["Movie not found"]

    idx = movies[movies["title_lower"] == movie_title].index[0]

    scores = list(enumerate(similarity[idx]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for i in scores[1:6]:
        recommendations.append(movies.iloc[i[0]]["title"])

    return recommendations