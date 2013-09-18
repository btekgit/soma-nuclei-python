# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 23:12:50 2013

@author: btek
"""

import numpy
import vigra
from FeatureSet import FeatureSet
from accuracyTools import calculateAccuracy
#from pylab import *
#import time #s,  datetime
from matplotlib import pylab







# this part has the main script!
# I hate this that there is no indicator. no divider mechanism
folder = '/mnt/hgfs/mouse_brain/20130506-interareal_mag4/ccout/whole_ilp8/'
#folder = '/mnt/hgfs/mouse_brain/tests/'
#featureFile = folder+'dtmask_th_50_a1000.h5_fea.h5'

featureFile = folder+'cc_th_50_detectionbb_mxlabel_all_regionProps.matcc_processed_th_1000_non_edge_matlab_fea.h5'
#featureFile = folder+'labels.h5_fea.h5'

labelFile = folder+'cc_th_50_detectionbb_mxlabel_all_regionProps.matcc_processed_th_1000_non_edge_training_labels.h5'
#labelFile = folder+'new.h5'
#trainingDataSetNames = ['count',
#'histogram', 
#'kurtosis',
#'maximum',
#'mean',
#'minimum',
#'quantiles',
#'regionAxes',
#'regionCenter',
#'regionRadii',
#'skewness']
#this is vigra computeed
#trainingDataSetNames = ['count', 'regionCenter', 'regionRadii', 'histogram']

# this is matlab 
trainingDataSetNames = ['Volume', 'CentroidNorm', 'Perimeter','Complexity',
 'BoundingBox2Volume','BoundingBoxAspectRatio', #, 'BoundingBoxAspectRatio',
   'IntensityHist']#'BoundingBoxAspectRatio']#, 'IntensityMax','IntensityMean',

numpy.random.seed(100)
print numpy.random.rand(1,1)
repeatN = 10
acc = numpy.zeros([repeatN, 2])

allFeatures = FeatureSet.readFromFile(featureFile, trainingDataSetNames,labelFile, 'labels')
coordinateFeatures = FeatureSet.readFromFile(featureFile,['Centroid'] ,labelFile, 'labels')
predThreshold = 0.5
thresholdRange = numpy.linspace(0.1, 0.999, 10)
roc_mn_acc=numpy.zeros([len(thresholdRange),2])
roc_std_acc=numpy.zeros([len(thresholdRange),2])

partreeCount = 50
counter = 0
for predThreshold in thresholdRange:
    for ite in range(0,repeatN):
        
       # trn, val = allFeatures.divideSetRandom(1,1,True)
        trn, val = allFeatures.divideSetByZ(coordinateFeatures.data[:,2])
        rf = vigra.learning.RandomForest(treeCount=partreeCount)
        rf.learnRF(trn.data, trn.labels)
    
        
        p = rf.predictProbabilities(val.data)
        predclasspositive = p[:,1]
        
        acc[ite,:] = calculateAccuracy(predclasspositive, predThreshold, val.labels)
    mn_acc = numpy.mean(acc,0)
    std_acc = numpy.std(acc,0)
    #print acc
    print "Mean acc = ", numpy.mean(acc,0), '\n'
    print "Std acc = ", numpy.std(acc,0),'\n'
    roc_mn_acc[counter,:] = mn_acc
    roc_std_acc[counter,:] = std_acc
    counter+=1
fig1 = pylab.figure(333)
pylab.plot(roc_mn_acc[:,1],roc_mn_acc[:,0])
pylab.hold(True)
pylab.show(fig1)
print "#################################"
print roc_mn_acc
print "#################################"
print roc_std_acc
    
