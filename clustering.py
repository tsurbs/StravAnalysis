from dataCleaning import dedensify
import numpy as np
if __name__ == "__main__":
    import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tcxParser as tp 
from JenksNaturalBreaks2D import JenksNaturalBreaks2D
import pickle as pkl
def kMeansN(n, resolution = 3):
    (Xs, Ys, Zs, Cs) = tp.getAllRunData()
    # with open("allRunData.pkl", "wb") as file: 
    #     pkl.dump((Xs, Ys, Zs), file)
    with open("allRunData.pkl", "rb") as file:
        (Xs, Ys, Zs, Cs) = pkl.load(file)
    print("data load")
    Xs, Ys = dedensify(Xs, Ys, Zs, Cs, resolution = resolution)
    print(Xs, Ys, len(Xs), "data clean")
    data = np.column_stack((Ys, Xs)) #zip but numpy
    kmeans = KMeans(n_clusters=n)

    kmeans.fit(data)

    labels = kmeans.labels_
   

    
    if __name__ == "__main__":
        # plt.scatter(Ys, Xs, c = labels)
        # plt.axis("equal")
        clusterXs = [a[0] for a in kmeans.cluster_centers_]
        clusterYs = [a[1] for a in kmeans.cluster_centers_]

        plt.scatter( Ys, Xs, c=labels)
        plt.scatter( clusterXs, clusterYs)

        plt.axis("equal")
        plt.show()
        
    return kmeans

def jnbN(n, resolution):
    # (Xs, Ys, Zs, Cs) = tp.getAllRunData()
    # print("data load")
    # with open("allRunData.pkl", "wb") as file: 
    #     pkl.dump((Xs, Ys, Zs, Cs), file)
    with open("allRunData.pkl", "rb") as file:
        (Xs, Ys, Zs, Cs) = pkl.load(file)

    print("data load")
    Xs, Ys = dedensify(Xs, Ys, Zs, Cs, resolution = resolution)
    print(Xs, Ys, len(Xs), "data clean")
    data = np.column_stack((Ys, Xs)) #zip but numpy
    data = np.column_stack((Ys, Xs)) #zip but numpy
    jenks = JenksNaturalBreaks2D(n)

    jenks.fit(data)

    labels = jenks.point_labels
    centers = jenks.label_centers

    
    if __name__ == "__main__":
        # plt.scatter(Ys, Xs, c = labels)
        # plt.axis("equal")
        clusterXs = [a[0] for a in centers]
        clusterYs = [a[1] for a in centers]

        plt.scatter( Ys, Xs, c=labels)
        plt.scatter( clusterXs, clusterYs)

        plt.axis("equal")
        plt.show()
    return jenks

clusters = kMeansN(300, 3)
with open("clusters.pkl", "wb") as file:
    pkl.dump(clusters, file)
