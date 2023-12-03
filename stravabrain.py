import os
import requests
import json

fakestdout = []

from sys import*;import io,os;V=argv;V.extend([""]*2);stdin=io.StringIO(V[2])if V[2]else stdin
r=os.path.exists(V[1]);f=r and open(V[1]);b=f.read()if r else V[1];r and f.close()
def I(b,t,p):
  while b: # interpret while there's code
    c,*b=b;c="+-><,.[]".find(c) # get next op
    if c in[0,1]:t[p]+=1-2*c;t[p]%=256 # increase memory cell and wrap at 256
    if c in[2,3]:p+=5-2*c;t=[0]*(p<0)+t+[0]*(p==len(t));p=max(p,0) # move pointer and adjust tape
    if c==4:i=stdin.read(1)or chr(0);t[p]=ord(i)%256 # read one char as numeric input
    if c==5:fakestdout.insert(0, chr(t[p])) # print one char as output
    if c==6:
      d=1;j=[d:=d+(x=="[")-(x=="]")for x in b].index(0);b,b_=b[j+1:],b[:j]
    while t[p]:t,p=I(b_,t,p) # loop while memory cell is non-zero
  return t,p

# t,p=I(b,[0],0);print();print(t,p) # interpret and print debugging info

refresh_data = {
    "client_id":os.getenv("CLIENT_ID"),
    "client_secret":os.getenv("AUTHSECRET"),
    "refresh_token":os.getenv("BEARERREFRESH"),
    "grant_type":"refresh_token"
}
refresh_response = requests.post("https://www.strava.com/oauth/token", data=refresh_data )
# print(refresh_response.json())
os.environ["BEARERACCESS"] = refresh_response.json()["access_token"]
headers = {
    'Authorization': 'Bearer ' + os.getenv('BEARERACCESS', ''),
}
activity_response = requests.get('https://www.strava.com/api/v3/athlete/activities', headers=headers)

most_recent_activity = activity_response.json()[0]

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
print(brainfuck_code)
try: 
    I(brainfuck_code, [0], 0)
    evaluated = "".join(fakestdout.reverse())
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
update_response = requests.put('https://www.strava.com/api/v3/activities/'+str(id), data=update_json, headers=headers)