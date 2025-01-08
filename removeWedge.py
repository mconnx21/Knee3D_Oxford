import numpy as np
from skimage import measure
from mhdMasks import write_ply_withNormals, pad
from matplotlib import pyplot as plt
import sys
import cv2
from pathlib import Path
import os
import math
np.set_printoptions(threshold = sys.maxsize)
from midpoint import linePixels2D

from readMLMasks import loadPatientMask

"""This file allows you to remove a wedge from the 3D mesh of a patients femure."""

#this pad the mask to double it in the z axis
#the doubling is approximate, and in real life should depend on the slice thickness
def newPad(arr):
    (x,y,z) = arr.shape
    paddedMask = np.ndarray((x,y,z*2))


    for i in range (0,x):
        currentSlice = arr[i]
        newSlice = np.ndarray((z*2, y))
        transp = np.transpose(currentSlice)
        for j in range (0,z):
            newSlice[2*j] = transp[j]
            newSlice[2*j+1] = transp[j]
        newSlice = np.transpose(newSlice)
        paddedMask[i] = newSlice
    return paddedMask




#PRE: angle 2 is bigger than angle1, both are less than pi/2
# POST: returns an array of dimensions volumeDeminsions, with 1s for pixels inside the wedge defined by lines going out at both angles from hingepoint, and 0 evrywhere else
# note that we assume the wedge triangle is defined looking down y towards +inf, with +ve x going down screen and +ve z going left to right
def generateOnePlaneWedge(hingePoint, angle1, angle2, volumeDimensions):
    wedgeVolume = np.ndarray(volumeDimensions, 'f')

    #we're going to assume angles aremeasured from halfline paralell to positive x axis coming from hingepoint

    #calcualte a direction vector for angle 1
    #make it big to reduce error, but still needs to be smaller than the actual size of the volume

    changeX1 = (-1 * math.sin(angle1)) 
    changeZ1 = math.cos(angle1)
    r1 = (-1*hingePoint[0])/changeX1
    k1 = (volumeDimensions[2] -hingePoint[2]) /changeZ1
    biggestScalarPoss1 = min(r1,k1)
    changeX1 = changeX1 * biggestScalarPoss1
    changeZ1 = changeZ1 * biggestScalarPoss1
    targetPoint1 = [hingePoint[0]+ changeX1, hingePoint[1], hingePoint[2]+ changeZ1]

    changeX2 = (-1 * math.sin(angle2))
    changeZ2 = math.cos(angle2)
    r2 = (-1*hingePoint[0])/changeX2
    k2 = (volumeDimensions[2] -hingePoint[2]) /changeZ2
    biggestScalarPoss2 = min(r2,k2)
    changeX2 = changeX2 * biggestScalarPoss2
    changeZ2 = changeZ2 * biggestScalarPoss2
    targetPoint2 = [hingePoint[0]+ changeX2, hingePoint[1], hingePoint[2]+ changeZ2]

    line1 = linePixels2D(hingePoint, targetPoint1)
    line2 = linePixels2D(hingePoint, targetPoint2)

    """
    xs1 = [point[0] for point in line1]
    zs1 = [point[1] for point in line1]
    xs2 = [point[0] for point in line2]
    zs2 = [point[1] for point in line2]
    print(line1)
    print(line2)

    plt.figure()

    plt.scatter(xs1, zs1)
    plt.scatter(xs2, zs2)
    plt.show()
    """

    x = hingePoint[0]
    i = 0
    j = 0
    while (x > line1[-1][0] and x > line2[-1][0]):
        #we expect line 2 to have a z value closer to 0  than lin1, for each x
        #so for line 2 we want to find the minimum z value with this x
        # and fir line 1 we want to find the maximum z value with this x
        #note every x value has an z value in both lines as the lines are not broken
        assert(line1[i][0] == x)
        while line1[i][0] == x:
            i+=1
        line1z = line1[i-1][1] 
        
        
        assert(line2[j][0] ==x)
        line2z = line2[j][1]
        while line2[j][0] == x:
            j+=1
        
        #now fill between the x values in this plane

        assert(line1z >= line2z)

        yrow = [0 for i in range (0, volumeDimensions[2])]
        for z in range(line2z, line1z+1):
            yrow[z] = 1.0
        wedgeVolume[x] = [yrow for i in range (0,volumeDimensions[1])]


        x-=1
    return wedgeVolume


#PRE: intersection point is a 3d coordinate. Angle is measured anticlockwise from vector going towards positive y. We define the halfline as the line starting at interseciton point and in direction theta
# we are definiing the line in the xy plane. I.e. when definiting the angle, set your axis up s.t. +ve y goes left to tight, +ve x goes down, and z goes into screem
#post: outputs a volume with 1s for every point to the right of the plane, 2s for every point on the half line and 0s everywhere else
def generateBiPlaneWedge(intersectionPoint, angle, volumeDimensions):
    assert(angle > (math.pi/2))

    cutVolume = np.ndarray(volumeDimensions, 'f')

    #we assume that the half line starts at intersection point and goes at direction angle measured from positive y direction
    #we want everything with an x value greater than intersectionPoint[0] to remain 0
    #we want everything to the right of the half line to get 1/2
    #and everything on the half line to get 1

    #first we pixelise the half line in the y-x plane

    changex = -1* math.sin(angle)
    changey = math.cos(angle)

    r = (-1*intersectionPoint[0])/changex
    k = (-1*intersectionPoint[1])/changey
    biggestScalarWeCanUse = min(r,k)

    changex = changex * biggestScalarWeCanUse
    changey = changey * biggestScalarWeCanUse
    targetx = intersectionPoint[0] + changex
    targety = intersectionPoint[1] + changey

    target = [targetx, intersectionPoint[2], targety]  ##THIS IS A MESS AS MIDPOINT WAS CODED TO USE X-Z AS THE 2D AXIS

    halfLine = linePixels2D([intersectionPoint[0], intersectionPoint[2], intersectionPoint[1]], target)

    x =  intersectionPoint[0]
    i = 0
    #rowOf1s = np.array([1.0 for i in range (0, volumeDimensions[2])])
    #rowOf2s = np.array([2.0 for i in range (0, volumeDimensions[2])])

    while (x > 0):
        #for each x want the biggest y , as it's for y values strictly greater than that y that we set the volume to have value 1/2
        assert(halfLine[i][0] == x)

        maxy = halfLine[i][1]
        

        while (halfLine[i][0] == x):
            for z in range(0, volumeDimensions[2]):
                cutVolume[x][halfLine[i][1]][z] = 2.0
            i+=1
        
        for y in range (maxy+1, volumeDimensions[1]):
            
            for z in range(0, volumeDimensions[2]):
                cutVolume[x][y][z] = 1.0


        x-=1

    #now deal with the x vlauyes higher than that of intersectionpoint i.e. points below the half line

    oppChangeX = math.sin(angle)
    oppChangeY = -math.cos(angle)
    r = (volumeDimensions[0]- intersectionPoint[0])/oppChangeX
    k = (volumeDimensions[1] - intersectionPoint[1])/oppChangeY
    biggestScalarWeCanUse = min(r,k)
    oppChangeX = oppChangeX*biggestScalarWeCanUse
    oppChangeY = oppChangeY*biggestScalarWeCanUse
    oppTarget = [intersectionPoint[0]+ oppChangeX, intersectionPoint[2], intersectionPoint[1] + oppChangeY]
    oppline = linePixels2D([intersectionPoint[0], intersectionPoint[2], intersectionPoint[1]], oppTarget)

    x = intersectionPoint[0]
    i = 0
    while (oppline[i][0] == x):
        i+=1
    x = oppline[i][0]

    while ( x< volumeDimensions[0]):
        assert( oppline[i][0] == x)
        while (oppline[i][0] == x):
            i+=1
        maxy = oppline[i-1][1]
        
        for y in range (maxy+1, volumeDimensions[1]):
            for z in range(0, volumeDimensions[2]):
                cutVolume[x][y][z] = 1.0
        
        x+=1
    

    return cutVolume




def combineWedge(biplanarwedge, singlewedge):
    added = np.add(biplanarwedge,singlewedge   ) 
    #note that now the points INSIDE THE NEW WEDGE THAT WE WANT, all have VALUE 2 or 3. 

    remove1s = (cv2.threshold(added, 1.0, 3.0, cv2.THRESH_TOZERO))[1] #descard values of 1
    normalise = remove1s.astype(bool) #turn the 2s and 3s to one so our array is boolean
    
    return normalise.astype(np.float32)

        

def writeRemovedWedgePLY(patient, side, visit, hingePoint, alpha1, alpha2, intersectionPoint, theta):

    folder, maskArray = loadPatientMask(patient, side, visit)
    femoralBone = (cv2.threshold(maskArray, 1, 7, cv2.THRESH_TOZERO_INV))[1] 
    femoralBone = np.array(femoralBone)
    femoralBone = newPad(femoralBone)
    paddedShape = femoralBone.shape
    cutVolume = generateBiPlaneWedge(intersectionPoint, theta, paddedShape)
    wedgeVolume = generateOnePlaneWedge(hingePoint, alpha1, alpha2,  paddedShape)
    combinedwedge = combineWedge(cutVolume, wedgeVolume)
    augmentedVolume = femoralBone + 10.0*combinedwedge
    subtractedVolume = (cv2.threshold(augmentedVolume, 7, 10, cv2.THRESH_TOZERO_INV))[1] #get rid of the wedge from femoral bone
    verts, faces, normals, values = measure.marching_cubes(subtractedVolume, 0) #for some reason need to do padding here unless you change pad funciton
    originalDirectory = os.getcwd()
    os.chdir(folder)
    filerootname = patient + "_" + side +"_"
    write_ply_withNormals(filerootname+"dewedged.ply", verts, normals)
    os.chdir(originalDirectory)

"""
#run the code on a real patient


patient = "9947240"
side = "LEFT"
visit = "1"
hingePoint = [130, 209, 35]
alpha1 = math.pi/8
alpha2 = 3*(math.pi/16)
intersectionPoint =[70,129,180]
theta = (2*math.pi)/3
writeRemovedWedgePLY(patient, side, visit, hingePoint, alpha1, alpha2, intersectionPoint, theta)
"""