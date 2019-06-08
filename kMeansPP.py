import numpy as np
import random
class kMeansPP :
    def __init__ (self, inputVecs) :
        self.inputs = inputVecs
        self.groupCenters = []
        self.__distance = []
        self.sse = 0
        
    def chooseInitCenter (self) :
        self.groupCenters.append(random.choice(self.inputs))
        
        return self
    
    def euclideanDistance (self) :
        self.sse = 0
        self.__distance = []
        groupCenters = np.array(self.groupCenters)
        
        for inputData in self.inputs :
            self.__distance.append(min(np.sqrt(np.sum(np.power(groupCenters - inputData, 2), axis = 1))))
            self.sse += self.__distance[-1]
            
        return self
    
    def chooseNewCenter (self) :
        probability = random.uniform(0, 1)
        self.sse *= probability
        
        for distanceNum in range(len(self.__distance)) :
            self.sse -= self.__distance[distanceNum]
            
            if self.sse < 0 :
                self.groupCenters.append(self.inputs[distanceNum])

                return self
    
    def run (self, centerNums) :
        self.chooseInitCenter()

        for _ in range(centerNums - 1) :
            self.euclideanDistance().chooseNewCenter()

        return self.groupCenters
