from readMLMasks import loadPatientMask
from angleViewer import *
import cv2
from matplotlib import pyplot as plt
import math

"""PURPOSE OF THIS SCRIPT:
#1. Define a better rotation function for mask images, which rotates incircle so that the ML algorithm cannot simply learn to track the image corners
#2. Use this to generate training data volumes which are all aligned to true coronal 
"""

sliceIncreaseCoefficient = 2

#data generation funciton, will pad the image out to have a circumcircile which then becomes the inscribed circle of a black bacground
#image should be cropped to square first, then circumcircled, then blacked.

def prepVolumeWithCircumCircleNew(mask):
    mask = mask[50:350] #crop

    #each base image we will make 300 x 300
    circumcircleRadius = int(math.sqrt(150**2 + 150**2))
    newHeight, newWidth = (circumcircleRadius *2,circumcircleRadius *2)
    newVolume =  np.ndarray((300, newHeight, newWidth))

    topLeft = (circumcircleRadius - 150, circumcircleRadius - 150)
    topRight = (circumcircleRadius - 150, circumcircleRadius + 150)
    bottomLeft = (circumcircleRadius + 150, circumcircleRadius - 150)
    bottomRight = (circumcircleRadius + 150, circumcircleRadius + 150)

    rowOfZeros = np.array([0 for i in range (0, newWidth)])

    
    #add the black background around the circumcircle
    for k in range (0,300):
        image = mask[k]
        image = np.where(image[0:300] !=0,1,0)
        newImage = np.ndarray((newHeight, newWidth))
        for y in range (0, topLeft[0]):
            newImage[y] = rowOfZeros
        for y in range (topLeft[0], bottomLeft[0]):
            for x in range (0, topLeft[1]):
                newImage[y][x] = 0
            for x in range (topLeft[1], topRight[1]):
                i , j= divmod((x- topLeft[1]),2)
                newImage[y][x] = image[y-topLeft[0]][i]
            for x in range (topRight[1], newWidth):
                newImage[y][x] = 0
        for y in range (bottomLeft[0], newHeight):
            newImage[y] = rowOfZeros
        newVolume[k] = newImage
    return newVolume



#rotate function:  given an image, I want to rotate the inscribed circle and place it back into the black square

def rotate_incircle(image, angle):
    # Ensure the image is square
    image = image.astype(np.uint8)
    height, width = image.shape
    assert height == width, "Image must be square"
    
    # Center and radius of the incircle
    center = (width // 2, height // 2)
    radius = width // 2
    
    # Create a circular mask for the incircle
    mask = np.zeros_like(image, dtype=np.uint8)
    
    mask = cv2.circle(mask, center, radius, 1, -1)

    # Isolate the incircle
    circle_region = cv2.bitwise_and(image, mask)
    
    # Create a rotated version of the circle region
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_circle = cv2.warpAffine(circle_region, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    
    # Mask the rotated circle to ensure only the incircle area is affected
    
    rotated_circle = cv2.bitwise_and(rotated_circle, mask)
    
    # Combine the rotated circle with a black background
    result = np.zeros_like(image, dtype=np.uint8)
    result = cv2.add(result, rotated_circle)
    
    return result

#assume patients[i] is patientAngles[i] away from true coronal, so we need to rotate by this angle to generate a true coronal volume
def generateTrueCoronalTrainingData(patients, patientAngles, filename):
    allImages= np.ndarray((len(patients)*300, 424,424))
    for i in range(0,len(patients)):
        patient = patients[i]
        _, mask = loadPatientMask(patient, "LEFT", "1")
        paddedMask = prepVolumeWithCircumCircleNew(mask)
        for j in range (0, 300):
            allImages[300*i + j] = rotate_incircle(paddedMask[j], patientAngles[i])
    
    currDir = os.curdir
    os.chdir("data")
    np.save(filename, allImages)
    os.chdir(currDir)



#patientAngles = [350, 353, 349, 353, 352, 358, 352, 350, 359, 357]
#patients = ["9911221", "9911721", "9912946", "9917307", "9918802", "9921811", "9924274", "9937239", "9938236", "9943227"]
#patientAngles = [350, 356, 351]
#patients = ["9947240", "9958234", "9964731"]
#patientAngles = [0, 0]
#patients = ["9947240", "9958234"]
#patients = ["9911221"]
#patientAngles = [350]
#generateTrueCoronalTrainingData(patients, patientAngles, "genuineUnseenOffsets.npy")



        



