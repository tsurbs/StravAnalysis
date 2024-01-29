import numpy as np
import random
import math
class JenksNaturalBreaks2D:
    data = np.array([])
    n_clusters = 0
    label_centers = np.array([])
    point_labels = np.array([])
    center_gradients = np.array([])
    points_at_centers = {}
    iter_number = 0

    def __init__(self, n_clusters = 7, data = []):
        self.n_clusters = n_clusters
        self.labels = self.fit(data)


    def fit(self, data):
        if len(data) == 0: return
        self.data = data if type(data) == type(np.ndarray) else np.array(data)

        # ((max X, max Y), (min X, min Y))
        bounds = ((np.max(data[:,0]), np.max(data[:,1])), (np.min(data[:,0]), np.min(data[:,1])))

        # initialize n_clusters random points
        self.label_centers = np.array([tuple([random.uniform(bounds[1][0], bounds[0][0]), random.uniform(bounds[1][1], bounds[0][1])]) for x in range(self.n_clusters)])
        self.point_labels = [self.getClosestCenter(p) for p in range(len(self.data))]
        
        while (self.iter_number < 5):

            for i in range(1):
                self.oneMeansMove()
            # Stopping here gives us kmeans
            print("Kmeans iter "+str(self.iter_number+1))

            self.calculateGradients()
            print("Calculated Gradients")

            movers = []
            for center, points in self.points_at_centers.items():
                if points is not None and len(points) > 0: 
                    movers += random.sample(points, math.ceil(len(points)/(self.iter_number + 2)))
            for pointIndex in movers:
                c1, c2 = self.getClosestCenter(pointIndex, get2Centers=True)
                if random.random() < self.center_gradients[self.point_labels[pointIndex]]:
                    if len(self.point_labels) != 0: # move point to new label
                        self.points_at_centers[self.point_labels[pointIndex]].remove(pointIndex)
                    if c2 not in self.points_at_centers:
                        self.points_at_centers[c2] = [pointIndex]
                    else: 
                        self.points_at_centers[c2].append(pointIndex)
                    self.point_labels[pointIndex] = c2
            
            print("iter", self.iter_number+1)
            self.iter_number += 1

    def oneMeansMove(self):        
        # reset centers
        for i in self.points_at_centers.keys():
            print("Centralizing region ", i)
            pointset = self.points_at_centers[i]
            new_x = sum([self.data[x, 0] for x in pointset])/len(pointset)
            new_y = sum([self.data[x, 1] for x in pointset])/len(pointset)
            self.label_centers[i] = tuple([new_x, new_y])
        self.point_labels = [self.getClosestCenter(p) for p in range(len(self.data))]
        

    def getClosestCenter(self, pIndex, get2Centers = False):
        point = [self.data[pIndex, 0], self.data[pIndex, 1]]
        closestDistance = None
        secondClosestDistance = None
        closestCenter = None 
        secondClosestCenter = None
        for i in range(len(self.label_centers)):
            center = [self.label_centers[i][0], self.label_centers[i][1]]
            dist = self.euclideanDistanceSq(point, center)
            if closestDistance == None: 
                closestCenter = i
                closestDistance = dist

            elif dist < closestDistance:
                    secondClosestCenter = closestCenter
                    secondClosestDistance = closestDistance
                    closestCenter = i
                    closestDistance = dist
            elif secondClosestCenter == None or dist > secondClosestDistance:
                    secondClosestCenter = i
                    secondClosestDistance = dist
        if not get2Centers:
            if len(self.point_labels) != 0: # move point to new label
                self.points_at_centers[self.point_labels[pIndex]].remove(pIndex)
            if closestCenter not in self.points_at_centers:
                self.points_at_centers[closestCenter] = [pIndex]
            else: 
                self.points_at_centers[closestCenter].append(pIndex)
        return [secondClosestCenter, closestCenter] if get2Centers else closestCenter
    
    
        

    def calculateGradients(self):
        means = []
        for i in self.points_at_centers.keys():
            label = self.label_centers[i]
            meanDist = 0.0
            for pIndex in self.points_at_centers[i]:
                point = [self.data[pIndex, 0], self.data[pIndex, 1]]
                dist = math.sqrt(self.euclideanDistanceSq(label, point))
                meanDist += dist / len(self.points_at_centers)
            means.append([meanDist, i])
        meanOfMeans = sum([m[0] for m in means])/len(means)
        meansCenteredLt = [(m - meanOfMeans, i) for m, i in means if m > meanOfMeans]
        meansCenteredLtAvg = sum([d[0] for d in meansCenteredLt])/len(meansCenteredLt)
        meansCenteredLtNormd = [(mc[0]/meansCenteredLtAvg, mc[1]) for mc in meansCenteredLt if mc[0] > meanOfMeans]
        meansCenteredGtr = [(m - meanOfMeans, i) for m, i in means if m <= meanOfMeans]
        meansCenteredGtrAvg = sum([d[0] for d in meansCenteredGtr])/len(meansCenteredGtr)
        meansCenteredGtrNormd = [(mc[0]/meansCenteredGtrAvg, mc[1]) for mc in meansCenteredGtr if mc[0] > meanOfMeans]
        self.center_gradients = np.zeros(self.n_clusters)
        for m, i in meansCenteredGtrNormd:
            self.center_gradients[i] = m 
        for m, i in meansCenteredLtNormd:
            self.center_gradients[i] = m 
            

    def euclideanDistanceSq(self, p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

