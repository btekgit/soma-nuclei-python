import numpy
import vigra
import h5py
import time #s,  datetime


class FeatureSet(object):
    @staticmethod
    def readFromFile(trainingFile, trainingDataSet,labelFile, labelDataSet):
    # label file is one
        print labelFile
        h5file = h5py.File(labelFile, 'r')
        h5data = h5file[labelDataSet]
        rowlength = len(h5data)
        
        #self.labels = numpy.uint32(numpy.squeeze(h5data[...]))
        labels = numpy.zeros([rowlength,1], dtype=numpy.uint32)
        labels[:,0] = numpy.uint32((h5data[...]))
#        print self.labels        
#        print "RowLength = rowlength", "Label Shape= ", numpy.shape(self.labels)
#        print "WARNING ROW LENGTH IS NOT CORRECT! FIX THIS"
        # more than one feature dataSet is merged into one data    
        # size is unknown so I will grow the array
        # data= []
        
        h5file = h5py.File(trainingFile)
        featureNames =numpy.copy(trainingDataSet)
        ix = 0
        for t in trainingDataSet:
            print "Feature DataSet", ix, "Name=", t, '\n'
            #time.sleep(2)
            # read dataset
            fDataSet= h5file[t]
            dsetShape = fDataSet.shape
            dsetDims = len(dsetShape)
            #compdim = dsetShape[dsetDims-1]
   #         print('warning ignoring label zero in the feature file')
            newfeaturecolumns =fDataSet[...,1:] 
            
            if(numpy.any(numpy.isnan(newfeaturecolumns))):
                print numpy.nonzero(numpy.isnan(newfeaturecolumns))
            
            fdim = newfeaturecolumns.ndim
            siz  = numpy.array(numpy.shape(newfeaturecolumns))                
            
            if(fdim ==1):
                 newfeaturecolumns = numpy.reshape(newfeaturecolumns, [siz[0],1])
                    
            
  #          print "dataSet Dims = ", fdim, ", Shape = ", siz,'\n'
            #if size is not correct transpose it
            
            if(fdim==2 & siz[0]!=rowlength):
                #print "transposing", siz, " shaped columns"
                newfeaturecolumns = numpy.transpose(newfeaturecolumns)
                siz = numpy.shape(newfeaturecolumns)
                #print "new size", siz, '\n'
            elif(fdim>2):
                #print "data shape", siz
                flatsize = numpy.prod(siz[siz!=rowlength])
                #print "reshaping", siz, " shaped columns into",rowlength,"x", flatsize,'\n'
                newfeaturecolumns = numpy.reshape(newfeaturecolumns, [rowlength,flatsize])
                siz = numpy.shape(newfeaturecolumns)
                #print "new size", siz, '\n'
            
            siz = numpy.shape(newfeaturecolumns)
            if (ix==0):
                data =  newfeaturecolumns
                featureIndexes = numpy.zeros([siz[0],1],'uint32')
                featureIndexes[:] = ix
            else:
#                print "Adding columns with shape=", numpy.shape(newfeaturecolumns)
 #               print " to Data shape=",numpy.shape(data),'\n'
                data = numpy.concatenate([data, newfeaturecolumns],axis=1)
                ty = numpy.ones([siz[0],siz[1]],'uint32')
                #print numpy.shape(featureIndexes)
                featureIndexes= numpy.concatenate([featureIndexes, ty], axis=1 )
            ix+= 1
            
            nanIx = numpy.isnan(newfeaturecolumns)
            if (numpy.any(nanIx)):
                print(t+' has NaNs')
            if (data.dtype!=numpy.float32):
                print("Warning converting data to float32")
                data= numpy.float32(data)
        
        sampleIndex = numpy.uint32(range(1, rowlength ))
        print "WARNING LABEL ZERO IS NOT READ!"
        return FeatureSet(data, labels, sampleIndex, featureIndexes, featureNames)
                
    def __init__(self, data, labels, sampleIndex=None,featureNames=None,featureIndexes=None ):
        # create structure
        self.data = data
        #self.index  
        self.labels = labels
        self.featureIndexes = featureIndexes
        self.featureNames = featureNames
        self.sampleNum = numpy.size(data,0)
        self.featureNum = numpy.size(data,1)
        self.sampleIndex = sampleIndex
      
        # this always refers to original configuration
        
        
        
    def divideSet(self, ratio1, ratio2,randomize=None):
        sampleIndex = self.sampleIndex
        sampleNum = len(sampleIndex) 
        firstPart = range(0, int(ratio1*sampleNum/float(ratio1+ratio2)))
        
        secondPart = range(int(ratio1*sampleNum/float(ratio1+ratio2)), sampleNum)
        if randomize is None:
            indexes =sampleIndex           
        else:
            indexes= numpy.random.permutation(sampleNum)
            indexes = sampleIndex[indexes]
        index1 = indexes[firstPart]       
        data1 = self.data[index1,:]
        label1 = self.labels[index1]
        
        index2 = indexes[secondPart]       
        data2 = self.data[index2,:]
        label2 = self.labels[index2]
        f1 = FeatureSet(data1,label1,index1, self.featureNames,self.featureIndexes)
        f2 = FeatureSet(data2,label2,index2, self.featureNames,self.featureIndexes)
        print "Input feature set of ", sampleNum, " rows is divided into two parts:", len(index1), " and ", len(index2), " samples" ,"\n"
        return f1,f2
        




### HERE MY FUNCTION FINISHES
def calculateAccuracy(pred,th, gt,labels=None):

    pred = numpy.squeeze(pred)
    gt = numpy.squeeze(gt)  
    if labels is None:
        labels = [0,1]
    positiveDT = pred>=th
    numPositiveDT = sum(positiveDT)
    
    positiveGT = gt==labels[1]
    negativeGT = gt==labels[0]
    numPositiveGT = sum(positiveGT)
    print "Positive GT =", numPositiveGT, "Negative GT = ", len(gt)-numPositiveGT
    print "Positive DT =", numPositiveDT, "Negative DT = ", len(pred)-numPositiveDT

    
    truePositive = (positiveDT == 1) & positiveGT
    sumTruePositive = sum(truePositive)
    falsePositive = (positiveDT == 1) & negativeGT
    sumFalsePositive = sum(falsePositive)
    print "True Positive =", sumTruePositive, "False Positive = ", sumFalsePositive
    results = numpy.array([sumTruePositive/float(numPositiveGT), sumFalsePositive/float(numPositiveDT)])
    return results




# this part has the main script!
# I hate this that there is no indicator. no divider mechanism
folder = '/mnt/hgfs/mouse_brain/20130506-interareal_mag4/ccout/whole_ilp8/'
#folder = '/mnt/hgfs/mouse_brain/tests/'
featureFile = folder+'dtmask_th_50_a1000.h5_fea.h5'
#featureFile = folder+'labels.h5_fea.h5'

labelFile = folder+'dtmask_th_50_a1000_labels.h5'
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

#trainingDataSetNames = ['count', 'regionCenter']
trainingDataSetNames = ['count', 'regionCenter', 'regionRadii', 'histogram']

numpy.random.seed(100)
print numpy.random.rand(1,1)
repeatN = 1
acc = numpy.zeros([repeatN, 2])
allFeatures = FeatureSet.readFromFile(featureFile, trainingDataSetNames,labelFile, 'labels')
for ite in range(0,repeatN):
    
    trn, val = allFeatures.divideSet(1,9,True)
    # parameters
    partreeCount = 10 
    
    rf = vigra.learning.RandomForest(treeCount=partreeCount)
    print "Type traindata = ", trn.data.dtype, "Type labels=", trn.labels.dtype,'\n'
    print "shape Train data = ", numpy.shape(trn.data), '\n'
    rf.learnRF(trn.data, trn.labels)
    #itefileName = labelFile+"forest_%.3d.h5" % ite
    #itefileName = labelFile+"forest_%.3d.h5" % ite
    #rf.writeHDF5(itefileName)
    
    #print "Shape Test Data=", numpy.shape(val.data), '\n'
    
    #print itefileName
    #rf = vigra.learning.RandomForest(itefileName)
    
    p = rf.predictProbabilities(val.data)
    predclasspositive = p[:,1]
    
    acc[ite,:] = calculateAccuracy(predclasspositive, 0.5, val.labels)
print acc
print "Mean acc = ", numpy.mean(acc,0), '\n'
print "Std acc = ", numpy.std(acc,0),'\n'



    
    
#    self.label = h5py.Dataset.read_direct()
##    def __init__(self,data, label, ix):
##        self.data = data
##        self.label = label
##        self.index = ix
#    def dummy(cls):
#        d = numpy.random.random((10,5)).astype(numpy.float32)
#        l = (numpy.random.random((10,1))+0.5).astype(numpy.uint32)
#        ix =numpy.array(range(1,10))
#        cls(FeatureSet(d,l,ix))
#        #print "there are", numpy.sum(l), " ones and ", (len(l)-numpy.sum(l))," zeros"
#        def fromFile()
            
#
#def readFeatureSet(trainingFile,trainingData):
#    
#    
#    set = FeatureSet
#    #set = FeatureSet(trainData,trainLabel, trainIndex)
#    return set
#    
