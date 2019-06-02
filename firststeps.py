# Import Pandas
import pandas as pd


# Function that computes the weighted rating of each movie
def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Main Function
def main():
    # Load Movies Metadata
    metadata = pd.read_csv('./movies/movies_metadata.csv', low_memory=False)

    # Calculate C
    C = metadata['vote_average'].mean()

    # Calculate the minimum number of votes required to be in the chart, m
    m = metadata['vote_count'].quantile(0.90)

    # Get the movies that have the minimum number of votes required
    q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

    # Calculate weighted rating of each movie
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1, m=m, C=C)

    # Sort movies based on score calculated above
    q_movies = q_movies.sort_values('score', ascending=False)

    # Print the top 15 movies based on IMDB formula
    print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))

if __name__ == '__main__':
    main()