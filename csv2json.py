import csv  
import json  
  
# Open the CSV  
f = open( 'movies/ratings_small.csv', 'rU' )  
# Change each fieldname to the appropriate field name. I know, so difficult.  
# reader = csv.DictReader( f, fieldnames = ( "adult","belongs_to_collection","budget","genres","homepage","id","imdb_id","original_language","original_title","overview","popularity","poster_path","production_companies","production_countries","release_date","revenue","runtime","spoken_languages","status","tagline","title","video","vote_average","vote_count" ))  
reader = csv.DictReader( f, fieldnames = ( "userId","movieId","rating","timestamp"))  
# Parse the CSV into JSON  
out = json.dumps( [ row for row in reader ] )  
print "JSON parsed!"  
# Save the JSON  
f = open( 'movies/ratings_small.json', 'w')  
f.write(out)  
print "JSON saved!"  
