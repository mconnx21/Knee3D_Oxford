
#let [0, depth] x[0,height] x [0,width] define a bounding box
#consider the circumcircle of the top square of the bounding box
# consider a camera on the circucicle, pointing towards the centre, in the direction of positive x  (so camera is positioned with a negative x coordinate)
# we will let this camera position be "position 0"
# we will measure angles anticlockwise round the circle, looking in direction of positive y
import math
from cohenSutherland import cohenSutherland
from midpoint import linePixels2D
from sliceBetweenPoints import loadDicom, fillGaps, displayASlice
import numpy as np
import matplotlib.pyplot as plt


"""This is old code no longer essential to the project, however it contains the original functions allowing
us to view any slice from any angle of the dicom volume. It is retained as it will be useful for thesis write up
to see the evolution of these functions."""

def padToCircle(volume):
    depth, height, width = volume.shape
    #want depth and width to be square
    if width == depth: 
        return volume
    elif depth > width:
        #pad width out to match depth
        numToPad = depth -width
        newVolume = np.ndarray((depth, height,depth), int)
        for i in range (0, depth):
            x = volume[i]
            for j in range(0, height):
                y = x[j]
                newRow = np.concatenate([y, np.array([0 for i in range(0, numToPad)])])
                newVolume[i][j] = newRow
        return newVolume
        
    else:
        #pad depth out to match width
        numToPad = width-depth
        paddingArrays = np.array([[[0 for i in range (0, width)] for j in range (0, height)] for j in range (0, numToPad)])
        newVolume = np.concatenate([volume, paddingArrays])
        return newVolume

        
#input: gaze angle realtive to "position 0", in radians, and bounding box dimensions
#output: locaation in real pixels of camera position at this angle, the location of the point on the circle opposite it (via diamater), and the direction vector from camera to centre
def getInfoFromAngle(theta,depth, height, width):
    #for now we will hardcode the position 0, but could add this as a paramater later
    assert(depth == width) #asserting its a circle at first
    circleCentre =  [depth/2, 0, width/2]
    print(circleCentre)
    #radius is magnitude of centre
    radius = math.sqrt(2*((depth/2)**2)) #=1/root(2)  * depth
    """omg huge issue, it's not a circle a all, it's an ellipse!  -- I can probs pad to circle with black squares -done"""
    #origin is on the circle, pi/4 radians anticlockwise from position 0
    #offsetAngle = theta - math.pi/4 #this is the angle from the origin rather than from position0
    newLocation = [int(round(circleCentre[0]- radius*math.cos(theta),0)),0, int(round(circleCentre[2]- radius*math.sin(theta),0))]
    oppLocation = [int(round(circleCentre[0]-radius*math.cos(theta+ math.pi),0)),0, int(round(circleCentre[2] - radius*math.sin(theta+math.pi) ,0))]
    dir = [oppLocation[0]- newLocation[0], oppLocation[1]-newLocation[1]]
    
    return newLocation, oppLocation, dir


#input: bounding box dimensions, two 2D points which theere is a line segement between, and y coordinate correpsonding to the plabne this line segment is in
#output: the two points on the bounding box which this lines intersectis
def intersectionWithBoundingBox(depth, height, width,point1, point2,y):
   intersection = cohenSutherland(depth-1, width-1, 0,0, point1, point2)
   #now need to turn these into integer pixels actually on the bounding box
   #well every intersection with the bouding box has at least one integer coordinate, and so rounding means we will always still be on the boundign box
   """since both points are outside the bounding box there is either 0 or two intersections"""
   if intersection == None:
       return None
   else:
       intersect1, intersect2 = intersection
       return [int(round(intersect1[0],0)), y, int(round(intersect1[1],0))], [int(round(intersect2[0],0)), y, int(round(intersect2[1],0))]


#for a given point and direction , calculate pixels of the line inside the top square of volume which passes through point and is perpendicular to direction
#misleading name, this is not the canera point, is a point on the line in direction of gaze of gamera
def findLine(cameraPoint, dir, volume):
    #need to find two distance points on the perpendicular line
    distanceMultiplier = 500
    perpDirVec = [dir[1], -dir[0]]
    point1 = [cameraPoint[0] + distanceMultiplier*perpDirVec[0], cameraPoint[2] + distanceMultiplier*perpDirVec[1] ]
    point2 = [cameraPoint[0] - distanceMultiplier*perpDirVec[0], cameraPoint[2] - distanceMultiplier*perpDirVec[1] ]
    depth,height, width = volume.shape
    intersection = intersectionWithBoundingBox(depth,height,width, point1, point2, cameraPoint[1])
    if intersection == None:
        return []
    else:
        intersect1, intersect2 = intersection
        line = linePixels2D(intersect1, intersect2)  #might get errors with intersection giving x value like on a graph, not an index
        return line

#put it all together to create the slices 
def generateSlices(theta, volume):
    depth, height, width = volume.shape
    cameraPosition, oppPosition, gaze = getInfoFromAngle(theta, depth, height, width)
    #xzCameraPos = [cameraPosition[0], cameraPosition[2]]
    #xzOppPos = [oppPosition[0], oppPosition[2]]
    print(cameraPosition, oppPosition)

    fig, ax = plt.subplots()
    #xs = [cameraPosition[0], oppPosition[0]]
    #ys = [cameraPosition[1], oppPosition[1]]
    #ax.scatter(xs, ys)
    slicePoints = linePixels2D(cameraPosition, oppPosition)
    #sliceXs = [p[0] for p in slicePoints]
    #sliceZs = [p[1] for p in slicePoints]
    #ax.scatter(sliceXs, sliceZs)
    #plt.show()

    #surroundedVolume = 

    slices = []
    n = len(slicePoints)
    maxSliceWidth = 0
    for i in range (0,n-1):
        point = slicePoints[i]
        #print(slicePoints)
        print(point)
        newPoint =[point[0],0, point[1]]
        topLine = findLine(newPoint, gaze, volume) #we essentially need to padd all slices with black s.t. the width of them all is the same as the width of the max top line
        if len(topLine) > maxSliceWidth:
            maxSliceWidth = len(topLine)
        thisSlice= []
        for y in range (0, height):
            thisRow = []
            for (x,z) in topLine:
                thisRow.append(volume[x][y][z])
            thisSlice.append(thisRow)
        slices.append(thisSlice)
    
    # now we have n-1 slices, each either height height but a different width
    #we now ened to pad them all
    #problem is - which sides to we pad. Hmmm. may need to do this as we go.
    #simpler solution: extend the volume all round to contain the circle. Make it black.
        
    return slices



DCMArrays, sliceThicknessSaggital, distanceBetweenZSaggital, distanceBetweenYSaggital, depth, height, width= loadDicom("oai\\Package_1216539_samples\\1.C.2\\9999865\\20060331\\11077102")
volume = fillGaps(DCMArrays, int(round( sliceThicknessSaggital/distanceBetweenZSaggital,0)))
circleVolume = padToCircle(volume)
theta  = math.pi/5
slices = generateSlices(theta, circleVolume)

for k in range (190, 193):
    displayASlice(slices[k])

            
#code is working with pi/4, though due to padding to make circle fit, you are actually looking at an angle which isn't the slice one that goes through the two corners of the original unpadded square
#fails on 90 degree angles due to being paralell with bounding boxes in cohen sutherland. Likely fixable bin indexes

#to work on next- output to cohen sutherland, indexes and ints vs floats