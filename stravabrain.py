import os
import requests
import json
import brainfuck

headers = {
    'Authorization': 'Bearer ' + os.getenv('BEARERACCESS', ''),
}

activity_response = requests.get('https://www.strava.com/api/v3/athlete/activities', headers=headers)
most_recent_activity = activity_response.json()[0]
print(most_recent_activity)
id = most_recent_activity['id']

# to get laps of most recent activity, replace "10273930982" with '+id+'
laps_response = requests.get('https://www.strava.com/api/v3/activities/'+str(id)+'/laps', headers=headers)
average_lap_speeds = [26.8224/(lap['average_speed']+.001) for lap in laps_response.json()]

def speed_to_bf_chrs(speed):
    if speed < 5.5: return ">"
    if speed < 6: return "<"
    if speed < 6.5: return "+"
    if speed < 7: return "-"
    if speed < 7.5: return "."
    if speed < 8: return ","
    if speed < 8.5: return "["
    if speed < 9: return "]"
    return ""

brainfuck_code = "".join([speed_to_bf_chrs(pace) for pace in average_lap_speeds])

try: 
    evaluated = brainfuck.evaluate(brainfuck_code)
except: 
    evaluated = "Runtime error :("

if evaluated == "": evaluated = "No output :("

update_json = {"description": "\n"+brainfuck_code+" || "+evaluated+""}

headers2 = {
    'Authorization': 'Bearer ' + os.getenv('BEARERACCESS', ''),
}
data = {
    "id":id,
    "data":update_json
}
# update_response = requests.put('https://www.strava.com/api/v3/activities/'+str(id), data=update_json, headers=headers)