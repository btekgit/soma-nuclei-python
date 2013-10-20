# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 01:29:44 2013

@author: btek
"""
import numpy

### HERE MY FUNCTION FINISHES
def calculateAccuracy(pred,th, gt,labels=None, verbose=False):

    pred = numpy.squeeze(pred)
    gt = numpy.squeeze(gt)  
    if labels is None:
        labels = [0,1]
    positiveDT = pred>=th
    numPositiveDT = sum(positiveDT)
    
    positiveGT = gt==labels[1]
    negativeGT = gt==labels[0]
    numPositiveGT = sum(positiveGT)
    if(verbose):
        print "Positive GT =", numPositiveGT, "Negative GT = ", len(gt)-numPositiveGT
        print "Positive DT =", numPositiveDT, "Negative DT = ", len(pred)-numPositiveDT

    
    truePositive = (positiveDT == 1) & positiveGT
    sumTruePositive = sum(truePositive)
    falsePositive = (positiveDT == 1) & negativeGT
    sumFalsePositive = sum(falsePositive)
    if(verbose):
        print "True Positive =", sumTruePositive, "False Positive = ", sumFalsePositive
    results = numpy.array([sumTruePositive/float(numPositiveGT), sumFalsePositive, sumTruePositive/float(numPositiveDT)])
    #results = numpy.array([sumTruePositive/float(numPositiveGT), sumFalsePositive])
    return results