import csv  
import json  
  
# Open the CSV  
f = open( 'movies/ratings_small.csv', 'rU' )  
# Change each fieldname to the appropriate field name. I know, so difficult.  
reader = csv.DictReader( f, fieldnames = ( "userId","movieId","rating","timestamp"))  
# Parse the CSV into JSON  
out = json.dumps( [ row for row in reader ] )  
print "JSON parsed!"  
# Save the JSON  
f = open( 'movies/ratings_small.json', 'w')  
f.write(out)  
print "JSON saved!"  
