import os
import requests
import datetime
import pickle as pkl
from tcxParser import getAllActivityIDs

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

def getRecentActivityIDs():
    btoken = getAccessToken()
    headers = {
        'Authorization': 'Bearer ' + btoken
    }
    activity_response = requests.get('https://www.strava.com/api/v3/athlete/activities/', headers=headers)

    return [x['id'] for x in activity_response.json()]

def loadTrainingData():
    if False:# if os.path.exists("ActivityDataList.pkl"):
    #     with open("ActivityDataList.pkl", "rb") as file:
    #         return pkl.load(file)
        pass
    else:
        activityNums = getAllActivityIDs()[-300:-150]

        ActivityDataList = [getActivityData(id) for id in activityNums]

        filtered = [t for t in ActivityDataList if t is not None]
        with open("ActivityDataList.pkl", "rb") as file:
            data = pkl.load(file)
        with open("ActivityDataList.pkl", "wb") as file:
            pkl.dump(data + filtered, file)

        # return filtered