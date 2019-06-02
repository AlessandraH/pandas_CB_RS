# Import Pandas
import pandas as pd
# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim, indices, metadata):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]


def main():
    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # Load Movies Metadata
    metadata = pd.read_csv('./movies/movies_metadata.csv', low_memory=False)


    # Replace NaN with an empty string
    metadata['overview'] = metadata['overview'].fillna('')

    # Calculate the minimum number of votes required to be in the chart, m
    m = metadata['vote_count'].quantile(0.40)

    # Get the movies that have the minimum number of votes required
    q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(q_movies['overview'])

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Configure q_movies to have all indexes
    q_movies = q_movies.reset_index(drop=True)

    # Construct a reverse map of indices and movie titles
    indices = pd.Series(q_movies.index, index=q_movies['title']).drop_duplicates()
    
    print(get_recommendations('The Dark Knight Rises', cosine_sim=cosine_sim, indices=indices, metadata=q_movies))
    print(get_recommendations('The Godfather', cosine_sim=cosine_sim, indices=indices, metadata=q_movies))

if __name__ == '__main__':
    main()
