from midpoint import linePixels2D

import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt

"""An old file no longer essential to the project, however it contains useful utilities for future work.
Allows you to choose two points on an axial cross section of a dicom volume and generate the vertical slcie passing though both those points """

#input path to dicoms
#output a 3D numpy array representing the volume of intensities, the slicethickness of the dicoms, and the pixel distances in each direction
def loadDicom(pathToDicom):
    DCMFiles = []
    DCMArrays = []

    for dirName, subdirList, fileList in os.walk(pathToDicom):
        for filename in fileList:
            thisFile = pydicom.dcmread(os.path.join(dirName, filename))
            DCMFiles.append(thisFile)
            for arr in pydicom.iter_pixels(thisFile): #this is a little hacky, only expecting generator to have 1 element
                
                DCMArrays.append(arr)

    sliceThicknessSaggital = float(DCMFiles[0].SliceThickness)
    pixelDistanceSaggital = (DCMFiles[0].PixelSpacing)  #this give this real world distance between centres of pixels going in z and y resp. Measured in mm
    distanceBetweenZSaggital, distanceBetweenYSaggital = pixelDistanceSaggital    #z goes accross pic like in define axis
    newSliceThickness = int(round(sliceThicknessSaggital / distanceBetweenZSaggital,0))
    #print(len(DCMFiles))
    DCMArrays = np.array(DCMArrays)
    depth, height, width = DCMArrays.shape
    return DCMArrays, sliceThicknessSaggital, distanceBetweenZSaggital, distanceBetweenYSaggital, depth, height, width

#input a 3D array of the pure dicom data volume, and the number of copies to make of each slice to fill slice gaps
#return a 3D numpy array of the volume with gaps filled
def fillGaps(DCMArrays, newSliceThickness):
    """going to load the files with duplicates corresponding to slice thickness, to make the cuboid proportionate"""
    depth, height, width = DCMArrays.shape
    paddedDepth = depth*newSliceThickness
    volume = np.ndarray((paddedDepth, height, width), int)
    x = 0
    j = 0
    while j < paddedDepth:
        for i in range (0, newSliceThickness):
            volume[j] = DCMArrays[x]
            j+=1
        x+=1
    return volume



#volume is a (paddedDepth , height, width) dimensioned volume

"""New stuff - going to attmept to get a slice at 45 degrees, from cupoint points [depth*saggitalSliceThickness, y, 0] and [0,y, width]"""
#input two 3D points whihc are PRECONDITIONED TO BE ON THE TOP SQUARE OF THE BOUNDING BOX 
#output a 2D array of intensities for the slice taken down y with those two points as the top corners
def sliceBetweenTwoPoints(volume, point1, point2, height):
    line = linePixels2D(point1, point2)
    thisSlice = []
    for y in range(0,height):
        thisRow = []
        for point in line:
            x,z = point
            intensity = volume[x][y][z]
            thisRow.append(intensity)
        thisSlice.append(thisRow)

    thisSlice = np.array(thisSlice)
    return thisSlice

def displayASlice(slice):
    fig, ax = plt.subplots()
    fig.set_facecolor('black')
    ax.imshow(slice, cmap ='gray', vmin = 0, vmax = 255)
    plt.show()


def showSlice(pathToDicom, point1, point2):
    DCMArrays, sliceThicknessSaggital, distanceBetweenZSaggital, distanceBetweenYSaggital, depth, height, width= loadDicom(pathToDicom)
    volume = fillGaps(DCMArrays, int(round( sliceThicknessSaggital/distanceBetweenZSaggital,0)))
    thisSlice = sliceBetweenTwoPoints(volume, point1, point2, height)
    displayASlice(thisSlice)


#showSlice("oai\\Package_1216539_samples\\1.C.2\\9999865\\20060331\\11077102", [307,0,0], [0,0,383])






