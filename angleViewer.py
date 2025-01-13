#let [0, depth] x[0,height] x [0,width] define a bounding box
#consider the circumcircle of the top square of the bounding box
# consider a camera on the circucicle, pointing towards the centre, in the direction of positive x  (so camera is positioned with a negative x coordinate)
# we will let this camera position be "position 0"
# we will measure angles anticlockwise round the circle, looking in direction of positive y

"""This code allows us to view all slices at any angle, when rotating around the axial axis.
Example usage is commented at the bottom."""

"""volume[d][h][w] is a pixel such that it's top left corner has coordinates (d, h, w) when working with geometry"""
import math
from cohenSutherland import cohenSutherland
from utilities import loadDicom, fillGaps, displayASlice
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

def generateCirclePixelsByDepth(centre, radius):
    #2*radius is the number of pixels in the diameter. So the centre lies between four neighbouring, and so is marked by the top left corner of the 4th quadrant pixel
    #will only do 4th quadrant then can work out the rest by symmetry as we go
    assert(radius>=1)
    diameter = 2*radius

    initialPixelOnCircle = [centre[0], centre[1] +radius]

    def f(x,z):
        P = (x-centre[0])**2 + (z-centre[1])**2 - radius**2
        return P

    assert (f(initialPixelOnCircle[0], initialPixelOnCircle[1] ) ==0)

    P = 0
    x,z = initialPixelOnCircle
    thisQuadrant = [[x,z]]
    cx, cz = centre
    depthBounds = [[z-diameter, z]]
    while(x< cx+radius):
        #options will be z-1, x+1  or z, x+1
        #midpoint to test is z-0.5, x+1
        if P +2*x -2*cx +1 -z +cz +0.25 <= 0:
            #then steeper midpoint is inside the circle
            #so want the steeper option
            P = P+2*x -2*cx +1
            x= x+1
            depthBounds.append([cz - (z-cz), z])
        #now test the shallower step
        #x, z-1
        elif P +x -cx + 0.25 -2*z +2*cz +1 >0:
            #then the shallow midpoint is outside so pick shallower step
            P = P -2*z +2*cz +1
            z = z-1
        else:
            P = P + 2*x -2*z - 2*centre[0] +2*centre[1] +2
            x= x+1
            z = z-1
            depthBounds.append([cz - (z-cz), z])

    #for the time being this gets the whole circle not skipping first and last depths
    depthBounds = np.array(depthBounds)

    fullCircle = np.concatenate( [depthBounds[::-1], depthBounds])
    xs =  np.concatenate([range(0,diameter+1), range(0,diameter+1)])

    """ys = [0 for i in range(0,diameter+diameter+2)]
    for x in xs:
        first, second = fullCircle[x]
        ys[x] = first
        ys[diameter +1 +x] = second
    plt.scatter(xs,ys)
    plt.show()"""

    truncatedCircle = fullCircle[1:diameter+1]

    
    return truncatedCircle #this is the full circle and by construction is guaranteed to have a pair of bounding pixels for every depth





def findCircumcircle(volume):
    depth, height, width = volume.shape
    #need distance from centre of top rectangle to corner
    return int(round((math.sqrt( (width/2)**2  + (depth/2)**2)),0))

#pad the volume such that it is a radius x height x radius volume, measured with top left corner at 0,0,0
#which contains volume centred ad the inscribed circle of the radius x radius top square
#precondition: radius is the radius of the circumcircle of the top face of volume
def padVolume(volume, radius):
    depth, height, width = volume.shape
    assert( width % 2 ==0 and depth % 2 == 0)
    depthDelta = int(radius - depth/2)
    widthDelta = int(radius - width/2)
    diameter = radius*2
    paddedVolume = np.ndarray((diameter, height, diameter), int)
    HWBlackSquare = [[0 for w in range(0, diameter)] for h in range(0,height)]
    toVolBlackRow = np.array([0 for w in range(0, widthDelta) ])
    for i in range (0, depthDelta):
        paddedVolume[i] = HWBlackSquare
    for d in range(0, depth):
        slice = []
        for y in range(0, height):
            thisRow = np.concatenate([toVolBlackRow, volume[d][y], toVolBlackRow])
            slice.append(thisRow)
        paddedVolume[depthDelta + d] = slice
    for i in range(0, depthDelta):
        paddedVolume[depthDelta + depth + i] = HWBlackSquare
    paddedVolume = np.array(paddedVolume)
    return paddedVolume

#input a volume of shapre depth x height x width , precondition is that depth == width

def circleBounds(volume):
    depth, height, width = volume.shape
    assert (depth ==width)
    diameter = depth
    assert(diameter %2 == 0)
    radius = int(depth/2 ) #radius of hte inscribed circle 
    centre = [radius, radius]  #be very careful with in terms of indexing vs coordinates. you should pass in the centre pixel 
    """top left corner of centre pixel is the geometric coordiantes of the centre of the circle"""

    circlePixelsByDepth = generateCirclePixelsByDepth(centre, radius)
    #circlePixelsByDepth[i] should be [a,b] where [i,a] and [i,b] (a<b) are the x-z coordinates of the two points at this depth which lie on the inscribed circle
    #i starts and 1 and ends at depth -2  so that we can unsure there are in fact two such points
    return circlePixelsByDepth

def camToGlobal(theta, point, centre):
    #point is a PIXEL in camera space, centre is a pixel in camera space given in 2D as the centre of the inscribed circle
    def translateCentreToOrigin(x,y,z):
        return [x-centre[0],y, z-centre[1]]
    def translateOriginToCentre(x,y,z):
        return [x+centre[0],y, z+centre[1]]
    def rotateClockPosY(x,y,z):
        #y remains the same
        return [ x*math.cos(-theta) + z*math.sin(-theta), y, -1*x*math.sin(-theta)+z*math.cos(-theta)]
    
    def findCameraPosition(y):
        x ,y, z = translateCentreToOrigin(0,y,0)
        x,y,z = rotateClockPosY(x,y,z)
        x,y,z = translateOriginToCentre(x,y,z)
        return [x,y,z]
    
    cx, cy, cz = findCameraPosition(point[1])
    whatHappens10 = rotateClockPosY(1,0,0)
    firstColumnRotateMatrix = [whatHappens10[0], whatHappens10[2]]
    whatHappens01 = rotateClockPosY(0,0,1)
    secondColumnRotateMatrix = [whatHappens01[0], whatHappens01[2]]
    x,y,z = point
    globalCoords = [firstColumnRotateMatrix[0]*x + secondColumnRotateMatrix[0] * z + cx, cy, firstColumnRotateMatrix[1]*x + secondColumnRotateMatrix[1]*z + cz]
    globalPixel = [int(round(globalCoords[0], 0)), int(round(globalCoords[1], 0)), int(round(globalCoords[2], 0))]
    return globalPixel


def generateSlices(theta, volume, start, stop, step):
    

    depth, height, width = volume.shape
    assert (depth == width)
    diameter = width
    assert(diameter %2 == 0)
    radius = int(diameter/2)
    circleBoundByDepth = circleBounds(volume)
    assert(len(circleBoundByDepth) == diameter) #==depth == width

    assert(stop <= diameter and start>=0)
    
    #assert (depth == width)
    slices = []
    for d in range(start,stop, step):  #change to 0 and depth for all slices
        print(d)
        #looking at slice d with respect to camera view
        thisSlice = []
        for y in range(0, height):
            a,b = circleBoundByDepth[d] 
            firstPointInCylinder = [d, a]
            secondPointInCylinder = [d, b]
            thisRow = np.ndarray(diameter, int)
            for z in range(0,a):
                thisRow[z] = 0
            for z in range(a,min(b+1,diameter)):
                gx, gy, gz = camToGlobal(theta,[d,y,z], [radius, radius]) #get global pixel of what the camera sees at dyz. input are point and centre of circle as PIXELS
                #print(gx,gy,gz)
                thisRow[z] = volume[min(gx,diameter-1)][gy][min(gz,diameter-1)]
            for z in range(b+1, diameter):
                thisRow[z] = 0
            thisSlice.append(thisRow)
        thisSlice = np.array(thisSlice)
        slices.append(thisSlice)
    slices = np.array(slices)
    #probably good to return it as same dimension volume that came in, so should add black swuare on either side
    #blackSquare = np.ndarray((height,diameter))
    #slices = np.concatenate([blackSquare, slices, blackSquare])
    #assert (slices.shape == (depth, height, width))
    return slices

patient1 = "oai\\Package_1216539_samples\\1.C.2\\9911221\\20050909\\10583703"
patient2 = "oai\\Package_1216539_samples\\1.C.2\\9911721\\20050812\\10478403"
patient3 = "oai\\Package_1216539_samples\\1.C.2\\9947240\\20051013\\10200003"

patient = patient3
patientID = patient[34: len(patient)]  #relies on the same directory structure as shown for the patients above

#print(patientID)

#INPUT: e.g. pi/2 radians, patient3, 1/3, 2/3, 50   to see patient 3's scan from an angle of pi/2, observing every 50th slice, ignoring the first and last third of slices.
#OUTPUT: displays images of the desired slices. If save= True then the images will be save din the angleViewer_slices directory.
def viewFromAngle(angle, patient, lowerFraction, upperFraction, step, save= False):
    
    DCMArrays, sliceThicknessSaggital, distanceBetweenZSaggital, distanceBetweenYSaggital, depth, height, width= loadDicom(patient)
    filledVolume = fillGaps(DCMArrays, int(round( sliceThicknessSaggital/distanceBetweenZSaggital,0)))
    #fvDepth, fvHeight, fvWidth = filledVolume.shape
    circumcirleRadius = findCircumcircle(filledVolume)
    #now going to padd this to a radius x fvHeight x radius volume, which contains the circumcylinder and is padded black around the filledVolume
    #padVolume expects a volume which has even dpeth and width - need to come back here and write a function to pad the filledVolume to even if not
    paddedvolume = padVolume(filledVolume, circumcirleRadius)
    #print(paddedvolume.shape)
    theta  = angle  #note -math.pi/2 + math.pi/16 is perfect elligned coronal for patient 3. -math.pi/2 is coronal which matches radiant viewe
    numberSlices = paddedvolume.shape[0]
    slices = generateSlices(theta, paddedvolume, int(lowerFraction*numberSlices), int(upperFraction*numberSlices), step)
    currentDir = os.curdir
    path = Path("angleViewer_slices\\"+patientID)  #save directory it will be better to later extend this to make subfolders based on angles
    path.mkdir(parents= True, exist_ok = True)
    os.chdir(path)

    for k in range (0, len(slices)):
        if save:
            cv2.imwrite(str(k) + ".png", slices[k])
        displayASlice(slices[k])
    os.chdir(currentDir)

#viewFromAngle(math.pi/14, patient, 1/3, 2/3, 50)
    
