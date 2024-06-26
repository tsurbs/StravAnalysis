from tcxParser import ezPlot
import treap
import numpy as np

def dedensify(Xs, Ys, Zs = None, Cs = None, resolution = 3, keepOrder = False):
    newPointsXs = treap.treap()
    for i in range(len(Xs)):
        x = round(Xs[i], resolution)
        y = round(Ys[i], resolution)

        if x not in newPointsXs:
            newPointsXs[x] = treap.treap()

        if y not in newPointsXs[x]:
            newPointsXs[x][y] = Zs[i] if Zs is not None else i
    
    Xs = []
    Ys = []

    for x in newPointsXs.keys():
        for y, z in newPointsXs[x].items():
            x = round(x, resolution)
            y = round(y, resolution)
            Xs.append((z, x))
            Ys.append((z, y))

    return (np.array([x[1] for x in sorted(Xs)]), np.array([y[1] for y in sorted(Ys)]))