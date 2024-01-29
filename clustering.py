import numpy as np
if __name__ == "__main__":
    import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tcxParser as tp 
from JenksNaturalBreaks2D import JenksNaturalBreaks2D
import pickle as pkl
def kMeansN(n):
    (Xs, Ys, Zs, Cs) = tp.getAllRunData()
    # with open("allRunData.pkl", "wb") as file: 
    #     pkl.dump((Xs, Ys, Zs), file)
    with open("allRunData.pkl", "rb") as file:
        (Xs, Ys, Zs) = pkl.load(file)
    print("data load")
    data = np.column_stack((Ys, Xs)) #zip but numpy
    kmeans = KMeans(n_clusters=n)

    kmeans.fit(data)

    labels = kmeans.labels_

    
    if __name__ == "__main__":
        plt.scatter(Ys, Xs, c = labels)
        plt.axis("equal")
        plt.show()
    return kmeans

def jnbN(n):
    # (Xs, Ys, Zs, Cs) = tp.getAllRunData()
    # print("data load")
    # with open("allRunData.pkl", "wb") as file: 
    #     pkl.dump((Xs, Ys, Zs, Cs), file)
    with open("allRunData.pkl", "rb") as file:
        (Xs, Ys, Zs, Cs) = pkl.load(file)
    print("data load")
    data = np.column_stack((Ys, Xs)) #zip but numpy
    jenks = JenksNaturalBreaks2D(n)

    jenks.fit(data)

    labels = jenks.point_labels

    
    if __name__ == "__main__":
        plt.scatter(Ys, Xs, c = labels)
        plt.axis("equal")
        plt.show()
    return jenks

clusters = jnbN(1500)
