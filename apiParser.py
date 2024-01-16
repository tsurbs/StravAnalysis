import os
import requests
import datetime
import json
def getAccessToken():
    if os.getenv("BEARERACCESS") is not None and datetime.datetime.now(datetime.timezone.utc).timestamp()+10 < int(os.getenv("access_exp_at")):
        return os.getenv("BEARERACCESS")
    refresh_data = {
        "client_id":os.getenv("CLIENT_ID"),
        "client_secret":os.getenv("AUTHSECRET"),
        "refresh_token":os.getenv("BEARERREFRESH"),
        "grant_type":"refresh_token"
    }

    refresh_response = requests.post("https://www.strava.com/oauth/token", data=refresh_data )

    os.environ["BEARERACCESS"] = refresh_response.json()["access_token"]
    os.environ["BEARERREFRESH"] = refresh_response.json()["refresh_token"]
    os.environ["access_exp_at"] = str(refresh_response.json()["expires_at"])

    return os.environ["BEARERACCESS"]

def getActivityData(activityId):
    btoken = getAccessToken()
    headers = {
        'Authorization': 'Bearer ' + btoken
    }
    print(headers, 'https://www.strava.com/api/v3/activities/'+str(activityId))
    activity_response = requests.get('https://www.strava.com/api/v3/activities/'+str(activityId), headers=headers)
    return activity_response.json()

def updateActivityData(activityId, update):

    btoken = getAccessToken()
    headers = {
        'Authorization': 'Bearer ' + btoken
    }
    return requests.put('https://www.strava.com/api/v3/activities/'+str(activityId), 
                        data=update, 
                        headers=headers
                        )

def getRecentActivityID(n_back = 0):
    btoken = getAccessToken()
    headers = {
        'Authorization': 'Bearer ' + btoken
    }
    activity_response = requests.get('https://www.strava.com/api/v3/athlete/activities', headers=headers)
    most_recent_activity = activity_response.json()[n_back]

    return most_recent_activity['id']