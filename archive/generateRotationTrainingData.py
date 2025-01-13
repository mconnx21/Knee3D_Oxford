from readMLMasks import loadPatientMask
import numpy as np
import pandas as pd
import os
from pathlib import Path
from matplotlib import pyplot as plt

"""PURPOSE OF THIS SCRIPT:
OUTDATED AND NEEDS TO BE CLEANED UP. Was orignally used to generate data for the rotnet on colab.
However, this uses the old rotate function instead of rotating the inscribed circle. 
newRotateForMasks is the newer version to use.
#This implements functions which load the ML masks, then crops, binarizes and pads them so that they can be fed into the ML RotNet
"""


#dimensions for the ML mask dataset is (364,364,150)  - this is all hardcoded and specific to this dataset
#this function is pretty much not used now that we have the version which includes slice thickness, but we keep it for this commit
def padToSquare(mask):
    mask = mask[50:350] #crop the mask so that all black slices aren't included
    reformedMask = np.ndarray((300, 250,250)) 
    for i in range (0, len(mask)):
        twoDmask = mask[i]
        #plt.imshow(twoDmask, cmap = 'gray')
        #plt.show()
        rows50black = np.array([[0 for i in range (0,250)]for j in range(0,50)])
        newSquareImage = twoDmask[50:300] #cropping in x direction
        #plt.imshow(newSquareImage, cmap = 'gray')
        #plt.show()
        twoDmaskT = np.transpose(newSquareImage)
        twoDmaskT = np.concatenate([rows50black, twoDmaskT, rows50black]) #padding in y direction
        newSquareImage = np.transpose(twoDmaskT)
        #plt.imshow(newSquareImage, cmap = 'gray')
        #plt.show()
        reformedMask[i] = newSquareImage
    return reformedMask

def padToSquareIncSliceThickness(mask):
    #for a 160 slice saggital view, we can assume that we (slicethickness, distancebetween pixels) = (0.7, 0.364)
    #and so the best approximation is to double every slice in the axial view
    #note that we don't know exactly which MRI was used for the masks
    #however it defintiely had at least 150 saggital slices
    #which is almost certainly going to end up needing a doubling (but no more) of slices to make a true axial view

    #right now we have 150 in axial view. Doubling this would make it 300 in size. We could take 300 in other direction too to make things simple, or could crop the axial a little
    #cropping looks hard so we will make 300x300 square images and deal with the slower training

    mask = mask[50:350]  #upper bound usually (50,350), this is just to debug
    reformedMask = np.ndarray((300, 300,300)) #first pararmeter usually 300, this is just to debug
    for i in range (0, len(mask)):
        twoDmask = mask[i][0:300]
        #plt.imshow(twoDmask, cmap = 'gray')
        #plt.show()
        
        newMask = np.ndarray((300,300))
        for row in range (0,300):
            for oldCol in range (0,150):
                newMask[row][oldCol*2] = twoDmask[row][oldCol]
                newMask[row][(oldCol*2)+1] = twoDmask[row][oldCol]
        #plt.imshow(newMask, cmap = 'gray')
        #plt.show()
        reformedMask[i] = newMask
    return reformedMask


def binarize(mask):
    binaryMask = np.where(mask!=0,1,0)
    return binaryMask


def generateTrainingData(patients, filename, padded = True):
    #assume side and visit are always left and 1 resp
    if padded:
        allPatients = np.ndarray((len(patients),300, 300,300))
    else:
        allPatients = np.ndarray((len(patients),300, 250,250))
    for i in range(0,len(patients)):
        patient = patients[i]
        _, mask = loadPatientMask(patient, "LEFT", "1")
        if padded:
            newMask = padToSquareIncSliceThickness(mask)
        else:

            newMask = padToSquare(mask)
        binaryMask = binarize(newMask)
        allPatients[i] = binaryMask
    allImages = np.concatenate(allPatients)
    currDir = os.curdir
    os.chdir("data")
    np.save(filename, allImages)
    os.chdir(currDir)



#patients = ["9911221", "9911721", "9912946", "9917307", "9918802", "9921811", "9924274", "9937239", "9938236", "9943227"]
#patients = ["9947240", "9958234", "9964731"]
#generateTrainingData(patients, 'axialImagesPaddedUnseen.npy')
#_, mask = loadPatientMask("9947240", "LEFT", "1")
#print(mask.shape)











