# Import Pandas
import pandas as pd
# Import CountVectorizer from scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
# Import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
# Parse the stringified features into their corresponding python objects
from ast import literal_eval
# Import Numpy 
import numpy as np

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

# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def main():
    # Load Movies Metadata
    metadata = pd.read_csv('./movies/movies_metadata.csv', low_memory=False)
    # Load keywords and credits
    credits = pd.read_csv('./movies/credits.csv')
    keywords = pd.read_csv('./movies/keywords.csv')

    # Remove rows with bad IDs.
    metadata = metadata.drop([19730, 29503, 35587])

    # Convert IDs to int. Required for merging
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    metadata['id'] = metadata['id'].astype('int')

    # Merge keywords and credits into your main metadata dataframe
    metadata = metadata.merge(credits, on='id')
    metadata = metadata.merge(keywords, on='id')

    # Replace NaN with an empty string
    metadata['overview'] = metadata['overview'].fillna('')

    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(literal_eval)

    # Define new director, cast, genres and keywords features that are in a suitable form.
    metadata['director'] = metadata['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(get_list)
    
    # Apply clean_data function to your features.
    features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)

    # Create a new soup feature
    metadata['soup'] = metadata.apply(create_soup, axis=1)


    # Calculate the minimum number of votes required to be in the chart, m
    m = metadata['vote_count'].quantile(0.40)

    # Get the movies that have the minimum number of votes required
    q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(q_movies['soup'])

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Configure q_movies to have all indexes
    q_movies = q_movies.reset_index(drop=True)

    # Construct a reverse map of indices and movie titles
    indices = pd.Series(q_movies.index, index=q_movies['title']).drop_duplicates()
    
    print(get_recommendations('The Dark Knight Rises', cosine_sim=cosine_sim, indices=indices, metadata=q_movies))
    print(get_recommendations('The Godfather', cosine_sim=cosine_sim, indices=indices, metadata=q_movies))
    print(get_recommendations(''))

if __name__ == '__main__':
    main()