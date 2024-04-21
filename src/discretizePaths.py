import pickle as pkl
from dataCleaning import dedensify
import tcxParser as tp 
import polyline
import matplotlib.pyplot as plt
import numpy as np

with open("clusters.pkl", "rb") as file:
    clusters = pkl.load(file)

def discPath(path, asPoints = False):
    Xs, Ys = path
    Xs, Ys = dedensify(Xs, Ys, resolution=4)
    data = np.column_stack((Ys, Xs)) #zip but numpy
    arr = clusters.predict(data)
    return [clusters.cluster_centers_[i] if asPoints else i for i in arr]


with open("activityDataList.pkl", "rb") as file:
        data = pkl.load(file)

print(len(data))
filtered = list(filter(lambda x: "map" in x and "polyline" in x["map"] and len(x["map"]["polyline"]) != 0, data))
mapped = list(map(lambda x: polyline.decode(x["map"]["polyline"]), filtered))

def isDiffClean(centers, asPoints = False):
    if asPoints:
        c1 = centers[0][1]
        for c in centers:
            print(c1, c[1])
            if c[1] != c1:
                return True 
        return False
    else: 
        c1 = centers[0]
        for c in centers:
            if c != c1:
                return True 
        return False

finalData = []

for i in range(len(mapped)):

    mXs = [p[0] for p in mapped[i]]
    mYs = [p[1] for p in mapped[i]]
    dp = discPath((mXs, mYs), asPoints = __name__ == "__main__")
    if isDiffClean(dp, asPoints = __name__ == "__main__"):
        finalData.append(dp)

        if __name__ == "__main__":
            Xs = [p[0] for p in dp]
            Ys = [p[1] for p in dp]
            
            clusterXs = [a[0] for a in clusters.cluster_centers_]
            clusterYs = [a[1] for a in clusters.cluster_centers_]

            # plt.scatter(clusterXs, clusterYs, )
            plt.scatter(Xs, Ys, c=plt.cm.get_cmap("hsv", len(mapped))(i))
plt.show()
