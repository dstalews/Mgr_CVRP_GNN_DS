import numpy as np
import random

class CVRPGenerator:
    """
    Create a CVRP dataset.
    """
### window_high set to 4000 to problems with long scheduling horizon
    def __init__(self, num_points, low=0, high=100):
        """
        Create a new CVRP dataset generator object.

        :param num_points: number of 2D data points to generate.
        :param num_vehicles: number of capacities to generate.
        :param low: lower bound of the range of coordinate space.
        :param high: upper bound of the range of coordinate space.
        """
        self._num_points = num_points
        self._low = low
        self._high = high
        self._c_low = 1
        self._c_high = 9

    def generate(self):
        """Generate a new CVRP dataset.

        This will generate a uniformly random matrix of shape (N, 2)
        representing N 2D coordinates for cities and a distance matrix for each
        of the points in the dataset

        :return: 2D dataset of city coordinates and a n by n distance matrix
        :rtype: (ndarray, ndarray)
        """
        cord = np.random.randint(self._low, self._high, size=(self._num_points, 2))

        demands = np.random.randint(self._c_low, self._c_high, size=self._num_points - 1)
        
        # x = [demands[i:i + 4] for i in range(0, len(demands), 4)]  

        # capacities = []
        # for l in x:
        #     capacities.append(sum(l))

        capacities = [15]*(round(self._num_points/3))
        random.shuffle(demands)
        demands = np.insert(demands, 0,0)
        return cord, demands, capacities
    
datasetCord = []
datasetDemands = []
datasetCapacities = []
size = 13

for i in range(0,20000):
     x = CVRPGenerator(num_points=size)
     cord, demands, capacities = x.generate()
     datasetCord.append(cord)
     datasetDemands.append(demands)
     datasetCapacities.append(capacities)

f = open("dataset/CVRP"+str(size)+".txt", "w")

for i in range(0,20000):
    f.write(str(size)+" "+str(len(capacities))+" ")
    for j in range(0,len(capacities)):
        f.write(str(datasetCapacities[i][j])+" ")
    for j in range(0,size):
        for z in range(0,2):
            f.write(str(datasetCord[i][j][z])+" ")
        f.write(str(datasetDemands[i][j])+" ")
    f.write("\n")
f.close()


datasetCord = []
datasetDemands = []
datasetCapacities = []
size = 25

for i in range(0,20000):
     x = CVRPGenerator(num_points=size)
     cord, demands, capacities = x.generate()
     datasetCord.append(cord)
     datasetDemands.append(demands)
     datasetCapacities.append(capacities)

f = open("dataset/CVRP"+str(size)+".txt", "w")

for i in range(0,20000):
    f.write(str(size)+" "+str(len(capacities))+" ")
    for j in range(0,len(capacities)):
        f.write(str(datasetCapacities[i][j])+" ")
    for j in range(0,size):
        for z in range(0,2):
            f.write(str(datasetCord[i][j][z])+" ")
        f.write(str(datasetDemands[i][j])+" ")
    f.write("\n")
f.close()

datasetCord = []
datasetDemands = []
datasetCapacities = []
size = 49

for i in range(0,20000):
     x = CVRPGenerator(num_points=size)
     cord, demands, capacities = x.generate()
     datasetCord.append(cord)
     datasetDemands.append(demands)
     datasetCapacities.append(capacities)

f = open("dataset/CVRP"+str(size)+".txt", "w")

for i in range(0,20000):
    f.write(str(size)+" "+str(len(capacities))+" ")
    for j in range(0,len(capacities)):
        f.write(str(datasetCapacities[i][j])+" ")
    for j in range(0,size):
        for z in range(0,2):
            f.write(str(datasetCord[i][j][z])+" ")
        f.write(str(datasetDemands[i][j])+" ")
    f.write("\n")
f.close()