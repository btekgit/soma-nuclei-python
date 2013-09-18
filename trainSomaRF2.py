import numpy
import vigra
import h5py
from FeatureSet import FeatureSet
from accuracyTools import calculateAccuracy
#from matplotlib import mlab 





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
   'IntensityHist'] #CloseMassRatio]#'BoundingBoxAspectRatio']#, 'IntensityMax','IntensityMean',

numpy.random.seed(100)
print numpy.random.rand(1,1)
repeatN = 100
acc = numpy.zeros([repeatN, 2])
allFeatures = FeatureSet.readFromFile(featureFile, trainingDataSetNames,labelFile, 'labels')
coordinateFeatures = FeatureSet.readFromFile(featureFile,['Centroid'] ,labelFile, 'labels')
predThreshold = 0.5
for ite in range(0,repeatN):
    
   # trn, val = allFeatures.divideSetRandom(1,1,True)
    trn, val = allFeatures.divideSetByZ(coordinateFeatures.data[:,2])
#    trn, val = allFeatures.divideSetByZ(allFeatures.data[:,allFeatures.featureNames=='regionCenter'])
    # parameters
    partreeCount = 50
    
    rf = vigra.learning.RandomForest(treeCount=partreeCount)
    #print "Type traindata = ", trn.data.dtype, "Type labels=", trn.labels.dtype,'\n'
    #print "shape Train data = ", numpy.shape(trn.data), '\n'
    rf.learnRF(trn.data, trn.labels)
    #itefileName = labelFile+"forest_%.3d.h5" % ite
    #itefileName = labelFile+"forest_%.3d.h5" % ite
    #rf.writeHDF5(itefileName)
    
    #print "Shape Test Data=", numpy.shape(val.data), '\n'
    
    #print itefileName
    #rf = vigra.learning.RandomForest(itefileName)
    
    p = rf.predictProbabilities(val.data)
    predclasspositive = p[:,1]
    
    acc[ite,:] = calculateAccuracy(predclasspositive, predThreshold, val.labels)
print acc
print "Mean acc = ", numpy.mean(acc,0), '\n'
print "Std acc = ", numpy.std(acc,0),'\n'
writeResults = 1
# calculate prediction for all
p = rf.predictProbabilities(allFeatures.data)
predclasspositive = p[:,1]
if(writeResults):
    f = h5py.File(featureFile+'_rf_results.h5','w')
    predLabels = predclasspositive>=predThreshold
    dset = f.create_dataset('rf_results',shape=numpy.shape(predLabels), dtype='uint32')
    dset[:] = predLabels
    dset.attrs.create('trainingFeatures',trainingDataSetNames)
    dset.attrs.create('predictionThreshold',predThreshold)
    dset.attrs.create('accuracy',acc)
