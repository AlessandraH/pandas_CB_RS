import json
from itertools import groupby

with open('movies/movies_metadata.json') as json_file:  
    movies = json.load(json_file)
    merged_ratings = []
    merged_ratings_list = []
    with open('movies/ratings_small.json') as json_file:  
        ratings = json.load(json_file)
        i = 0
        for r in ratings:
            i = i + 1
            print(i, "/", len(ratings))
            movielist = list(filter(lambda movie: movie["id"] == r["movieId"], movies))
            if len(movielist) > 0:
                r['movie_title'] =  movielist[0]['title']
                merged_ratings.append(r)
            print('')

        for k,v in groupby(merged_ratings,key=lambda x:x['userId']):
            userlist = list(v)
            if(len(userlist) >= 10):
                userlist = sorted(userlist, key = lambda i: i['rating'],reverse=True) 
                merged_ratings_list.append(userlist)

        with open('movies/merged_ratings.json', 'w') as outfile:  
            json.dump(merged_ratings_list, outfile)