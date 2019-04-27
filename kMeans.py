import numpy as np
class kMeans () :
    def __init__ (self, inputVec, groupNum, convergence = 0.001) :
        self.inputs = inputVec
        self.groupNum = groupNum
        self.convergence = convergence
        self.groups = []
        self.oldCenters = np.array([])
        self.groupCenters = np.array([])
        

    def setParams (self, inputVec, groupNum, convergence = 0.001) :
        self.inputs = inputVec
        self.groupNum = groupNum
        self.convergence = convergence

    def initGroupCenters (self, rang, dimensions) :
        import random
        lower = rang[0]
        upper = rang[1]
        
        self.groupCenters = np.array([[random.uniform(lower, upper) for i in range(dimensions)] for j in range(self.groupNum)])
    
        return self

    def cluster (self) :
        self.groups = [[] for i in range(self.groupNum)]
        
        for inputData in self.inputs :
            __distance = np.array([])
            __distance = (np.sum (np.sqrt (np.power (self.groupCenters - inputData, 2)), axis = 1))
            
            self.groups[np.argmin(__distance)].append (inputData)
        
        return self

    def updateCenters (self) :
        self.oldCenters = self.groupCenters.copy()
        
        for nowGroup in range(len(self.groupCenters)) :
            group = np.array(self.groups[nowGroup])

            self.groupCenters[nowGroup] = np.sum(group, axis = 0) / len(group)

        return self

    def evaluation (self) :
        error = sum(np.sqrt(np.sum(np.power(self.oldCenters - self.groupCenters, 2), axis = 1)))
        if error < self.convergence :
            return True

        return False
