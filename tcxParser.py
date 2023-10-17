import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

tree = ET.parse('test_run.tcx')
root = tree.getroot()
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

def getFeature(feature, tree):
    root = tree.getroot()
    l = list(map(lambda E: float(E.text), getAllTagsDFS(feature, root)))
    Fs = np.array(l)
    return Fs
cmap = plt.get_cmap('spring', 100)
def plotARun(tree, focusFeature = None):
    Xs, Ys = getXY(tree)
    Fs = getFeature(focusFeature, tree)
    plt.plot(Xs, Ys)
    plt.show()


plotARun(tree, "}Speed")