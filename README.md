# StravAnalysis

This is a project where I'm just playing around with the data from my strava runs to see what I can do with it.  For obvious security reasons, I don't plan to include the source data files, though I might later on(after all the data is already public on strava).  Following is a todo list of ideas:

 - [x] Get map data from tcx files, map with mpl
 - [x] Get $\alpha$ data from tcx files, plot with locn data
 - [ ] Identify turns
 - [ ] Identify segments
 - [ ] Create routes from constituient segments
 - [ ] Analyze pace performance given other factors
 - [ ] Create routes to maximize performance given known factors
 - [ ] Do sentiment analysis of post words
 - [ ] Create routes to maximize sentiment
 - [ ] Generate good routes in a place with no data
 - [ ] Generate commentary given run data

## API Side: EsoLang
curl -X POST https://www.strava.com/oauth/token \
	-F client_id=$CLIENT_ID\
	-F client_secret=$AUTHSECRET \
	-F code=$AUTHCODE \
	-F grant_type=authorization_code

curl --location 'https://www.strava.com/api/v3/athlete/activities' \
 --header 'Authorization: Bearer $BEARER_ACCESS'