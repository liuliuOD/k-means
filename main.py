import numpy as np
import random
from kMeans import kMeans
from kMeansPP import kMeansPP
from modules.visualize import visualize
import numpy as np

color = ['r', 'g', 'b', 'y', 'm']
centerColor = ['black', 'brown', 'gray', 'green', 'cyan']
centerShape = ['1', 'x', 6, '1', 'x']
centerNum = 5
dimensions = 2
dataNum = 10000
iterate = 1000

data = [[random.uniform(1, 10) for i in range(dimensions)] for j in range(dataNum)]
test = kMeans(data, centerNum)
drawer = visualize()
#test.initGroupCenters((1.0, 10.0), dimensions)
kmpp = kMeansPP(data)

test.setCenters(np.asarray(kmpp.run(centerNum)))

drawer.interactive()
for i in range(iterate) :
    test.cluster().updateCenters()

    for j in range(centerNum) :
        tmp = np.array(test.groups[j])
        drawer.scatter(tmp[:, 0], tmp[:, 1], color = color[j])

    for r in range(centerNum) :
        tmp = test.groupCenters[r]
        drawer.scatter(tmp[0], tmp[1], color = centerColor[r], marker = centerShape[r], size = 200)
    
    drawer.draw().pause(0.1).clearWindow()
    
    if test.evaluation() :
        break
