from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# suppose we have the following dataset of movies and their descriptions
movies = {
    "movie1": "Action Adventure Fantasy",
    "movie2": "Drama Romance War",
    "movie3": "Adventure Drama Fantasy",
    "movie4": "Crime Drama",
    "movie5": "Action Adventure Romance"
}

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the data
tfidf_matrix = vectorizer.fit_transform(movies.values())

# Calculate the cosine similarity of every item with every other item
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get the most similar items
def get_recommendations(title):
    idx = list(movies.keys()).index(title)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4] # Get the 3 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    return [list(movies.keys())[i] for i in movie_indices]

# Get recommendations for 'movie1'
print(get_recommendations('movie1'))