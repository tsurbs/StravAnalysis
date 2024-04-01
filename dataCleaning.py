from tcxParser import ezPlot
import treap
import numpy as np

def dedensify(Xs, Ys, Zs = None, Cs = None, resolution = 3):
    newPointsXs = treap.treap()
    for i in range(len(Xs)):
        x = round(Xs[i], resolution)
        y = round(Ys[i], resolution)

        if x not in newPointsXs:
            newPointsXs[x] = treap.treap()

        if y not in newPointsXs[x]:
            newPointsXs[x][y] = Zs[i] if Zs is not None else 1
    
    Xs = []
    Ys = []

    for x in newPointsXs.keys():
        for y in newPointsXs[x].keys():
            x = round(x, resolution)
            y = round(y, resolution)
            Xs.append(x)
            Ys.append(y)

    return (np.array(Xs), np.array(Ys))