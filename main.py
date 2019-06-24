# Import Pandas
import pandas as pd
# Import CountVectorizer from scikit-learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
# Parse the stringified features into their corresponding python objects
from ast import literal_eval
# Import Numpy 
import numpy as np
# Import JSON
import json

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim, indices, metadata):
    try:
        # Get the index of the movie that matches the title
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:6]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 5 most similar movies
        return metadata['title'].iloc[movie_indices].tolist()
    except:
        return []

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

def load_ratings(path):
    with open(path) as json_file:  
            ratings = json.load(json_file)
            return ratings

def create_folds(ratings):
    ratings_folds = list()
    for r in ratings:
        rating_folds = list()
        for i in range(10):
            rating_folds.append(list())
        ctrl = 0        
        while ctrl < len(r):
            rating_folds[ctrl%10].append(r[ctrl])
            ctrl = ctrl + 1        
        ratings_folds.append(rating_folds)
    return ratings_folds


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
    m = metadata['vote_count'].quantile(0.3)

    # Get the movies that have the minimum number of votes required
    q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

    """ Uncomment for Count """
    count = CountVectorizer(stop_words='english')
    """ Uncomment for TF-IDF """
    # count = TfidfVectorizer(stop_words='english')
    count_matrix = count.fit_transform(q_movies['soup'])

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Configure q_movies to have all indexes
    q_movies = q_movies.reset_index(drop=True)

    # Construct a reverse map of indices and movie titles
    indices = pd.Series(q_movies.index, index=q_movies['title']).drop_duplicates()

    ratings = load_ratings(path='movies/merged_ratings.json')
    ratings = create_folds(ratings)

    evaluations = list()

    # For each user
    for r in ratings:
        evaluation = {
            'userId': r[0][0]['userId']
        }
        # For each test to be done
        for test in range(10):
            train = list()

            # For each fold
            for i in range(10):
                # If fold isnt test
                if i != test:
                    # For each movie in fold
                    for m in r[i]:
                        train.append(m["movie_title"])

            recomendations = list()
            for t in train:
                recomendations = recomendations  + get_recommendations(t, cosine_sim=cosine_sim, indices=indices, metadata=q_movies)
            print("user", r[0][0]['userId'], "fold", test+1)
            evaluation[str(test+1)] = len(set([d['movie_title'] for d in r[test]]).intersection(set(recomendations)))/len(r[test])
        evaluations.append(evaluation)
    with open('movies/evaluation.json', 'w') as outfile:  
            json.dump(evaluations, outfile)
    # print(get_recommendations('10 Things I Hate About You', cosine_sim=cosine_sim, indices=indices, metadata=q_movies))


if __name__ == '__main__':
    main()
