import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import os 
import csv 
# tree = ET.parse('The_Data/abc.tcx')
# root = tree.getroot()
def expandToDepthD(root, d, showDepth = False, showTag = False, showText = True):
    return [] if d == 0 else [[] + [d] if showDepth else [] + [child.tag] if showTag else [] + [child.text] if showText else []  + expandToDepthD(child, d-1) for child in root]
# print(expandToDepthD(root[0][0][1], 1))

def getFirstTagDFS(tagend, root):
    for s in root:
        if s.tag.endswith(tagend):
            return s  
        else: 
            ret = getFirstTagDFS(tagend, s)
            if ret is not None: return ret

def getAllTagsDFS(tagend, root, init = []):
    for s in root:
        if s.tag.endswith(tagend):
            init = getAllTagsDFS(tagend, s, init + [s])
        else: 
            init = getAllTagsDFS(tagend, s, init)
    return init

def getTotalTimeS(tree):
    root = tree.getroot()
    return root[0][0][1][0].text

def getTotalDistanceM(tree):
    root = tree.getroot()
    return root[0][0][1][1].text

def getPaceMinMi(tree):
    return (float(getTotalTimeS(tree))/60)/(float(getTotalDistanceM(tree))/1609)

#TODO: do this with numpy
def getLocnsList(tree):
    root = tree.getroot()
    return list(map(lambda x: expandToDepthD(x, 1), getAllTagsDFS("Position", root)))


def getXY(tree):
    xyPairs = getLocnsList(tree)
    Xs = np.array(list(map(lambda xy: float(xy[0][0]), xyPairs)))
    Ys = np.array(list(map(lambda xy: float(xy[1][0]), xyPairs)))
    return (Xs, Ys)

def getFeature(feature, tree, asColor = True):
    root = tree.getroot()
    l = list(map(lambda E: float(E.text), getAllTagsDFS(feature, root)))
    Fs = np.array(l)
    return Fs

def ezPlot(Xs, Ys, Zs = None, colors = None, threeD = False):
    if colors is None: colors = np.zeros(len(Xs))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') if threeD else fig.add_subplot()
    ax.scatter(Xs, Ys, Zs, c = colors) if threeD else ax.scatter(Xs, Ys, c = colors)
    plt.show()

def plotARun(tree, focusFeature = None):
    Xs, Ys = getXY(tree)
    Zs = getFeature("}AltitudeMeters", tree, asColor=False)
    colors = getFeature(focusFeature, tree, asColor=True)
    _min = min(np.shape(Xs)[0], np.shape(Ys)[0], np.shape(Zs)[0], np.shape(colors)[0])
    Xs = Xs[:_min]
    Ys = Ys[:_min]
    Zs = Zs[:_min]
    colors = colors[:_min]
    return (Xs, Ys, Zs, colors)


# (Xs, Ys, Zs, Cs) = plotARun(ET.parse('The_Data/5717539530.tcx') , "}Speed")


def detectCurves(tree, resolution = 5):
    def slope(x1, y1, x2, y2):
        if x1 == x2: return 0
        elif x1>x2:
            return (y1-y2)/(x1-x2)
        return (y2-y1)/(x2-x1)
    def stDev(list):
        mean = sum(list) / len(list) 
        variance = sum([((x - mean) ** 2) for x in list]) / len(list) 
        return variance ** 0.5

    Xs, Ys = getXY(tree)
    Zs = getFeature("}AltitudeMeters", tree, asColor=False)
    curvinessAsFeature = np.zeros(len(Xs))
    for i in range((resolution//4), len(Xs)-(resolution//4)):
        slopeAdd = stDev([slope(Xs[i], Ys[i], Xs[i+j], Ys[i+j]) for j in range(-resolution//2, (resolution)//2-1)])
        print([slope(Xs[i], Ys[i], Xs[i+j], Ys[i+j]) for j in range(-resolution//2, (resolution)//2-1)])
        curvinessAsFeature[i] += slopeAdd
    print(curvinessAsFeature)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(Xs, Ys, c = curvinessAsFeature)
    plt.show()
# detectCurves(tree, 9)

def isRunIn(tree, x1, y1, x2, y2):
    xMin, xMax = (min(x1, x2), max(x1, x2))
    yMin, yMax = (min(y1, y2), max(y1, y2))
    root = tree.getroot()
    node = getFirstTagDFS("Position", root)
    if node is None: return False
    firstXY = expandToDepthD(node, 1)
    return xMin < float(firstXY[0][0]) < xMax and yMin < float(firstXY[1][0]) < yMax

def getAllActivityIDs():
    with open("Raw_Strava_Data/activities.csv", newline="") as file:
        a = csv.reader(file)
        return [r[0] for r in a][1:]


def getAllRunData():
    allFiles = os.listdir("./Raw_Strava_Data/activities")
    allXs = np.empty(0)
    allYs = np.empty(0)
    allZs = np.empty(0)
    allCs = np.empty(0)
    for f in allFiles:
        try: 
            tree = ET.parse('./Raw_Strava_Data/activities/'+f) 
        except ET.ParseError: pass

    #40.43798619868844, -79.97414138202551, 40.48980414288396, -79.9924162455036

        if isRunIn(tree,40.376922439181726, -79.82669782619318, 40.48980414288396, -79.9924162455036): 
            (Xs, Ys, Zs, colors) = plotARun(tree, "}Speed")
            allXs = np.concatenate([allXs, Xs])
            allYs = np.concatenate([allYs, Ys])
            allZs = np.concatenate([allZs, Zs])
            allCs = np.concatenate([allCs, colors])
    # ezPlot(allXs, allYs, allZs, allZs, threeD=False)
    return (allXs, allYs, allZs, allCs)

