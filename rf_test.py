import numpy
import vigra

d = numpy.random.random((200,5)).astype(numpy.float32)
l = (numpy.random.random((200,1))+0.5).astype(numpy.uint32)
print "there are", numpy.sum(l), " ones and ", (len(l)-numpy.sum(l))," zeros"

rf = vigra.learning.RandomForest(treeCount=10)
rf.learnRF(d, l)
rf.writeHDF5("my_forest.h5")

rf2 = vigra.learning.RandomForest("my_forest.h5")

d2 = numpy.random.random((1000,5)).astype(numpy.float32)
p = rf2.predictProbabilities(d2)
print p.shape