import numpy as np
if __name__ == "__main__":
    import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tcxParser as tp 
def kMeansN(n):
    (Xs, Ys, Zs, Cs) = tp.getAllRunData()

    data = np.column_stack((Xs, Ys)) #zip but numpy
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data)
    labels = kmeans.labels_

    
    if __name__ == "__main__":
        plt.scatter(Ys, Xs, c = labels)
        plt.axis("equal")
        plt.show()
    return Xs, Ys, labels
kMeansN(10)