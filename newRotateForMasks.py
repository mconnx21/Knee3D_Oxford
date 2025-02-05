from readMLMasks import loadPatientMask
from angleViewer import *
import cv2
from matplotlib import pyplot as plt
import math
#from MRI_to_Xray import MRI_to_Xray

"""PURPOSE OF THIS SCRIPT:
#1. Define a better rotation function for mask images, which rotates incircle so that the ML algorithm cannot simply learn to track the image corners
#2. Use this to generate training data volumes which are all aligned to true coronal 
#This is specifically designed for the ML masks given  here: 
# And is not transferrable 
"""

sliceIncreaseCoefficient = 2

#data generation funciton, will pad the image out to have a circumcircile which then becomes the inscribed circle of a black bacground
#image should be cropped to square first, then circumcircled, then blacked.

def prepVolumeWithCircumCircleNew(mask):

    mask = mask[50:350] #take 300 axial slices for each mask
    """
    #Alternative option was 256 x256 squares, however the knee data won't all fit within the incircle of such a size. It is a pretty tight cropping of the knees though

    #each slice image we will make y x z = 256 x 256
    #For y axis we simply crop 20:276
    #For z axis , we first need to add 2 copies of each pixel going down z axis to make proportional (filling in slice thickness)
    #Then we will shave off 22 pixels from each
    #Equivilently, we can shave the first and last 11 pixels off, and then double every pixel

    
    newHeight, newWidth = (256,256)
    newVolume =  np.ndarray((300, newHeight, newWidth))
    for k in range (0,300):
        image = mask[k]
        image = np.where(image[0:300] !=0,1,0)  #binarize the image to not distinguish between different types of knee tissue
        newImage = np.ndarray((newHeight, newWidth))
        for y in range (0, 256):
            row = image[y+20]  # we will take rows form the original slice in [20:276]
            row = row[11: (len(row)-11)] #shave off 11 z-pixels eeither side to make row length 128
            for z in range (0,128):   #now essentially double each z to make row length 256
                newImage[y][2*z] = row[z]
                newImage[y][(2*z +1)] = row[z]
        newVolume[k] = newImage
    return newVolume
    """
    
    
    #OPTION 2: origional approach which produces 424x424 slices
    #we make a circumcircle around where the knee data is in the image, and padd the image to fit the circumcircle as an incircle
    
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
        image = np.where(image[0:300] !=0,1,0)   #could change the range here if we wanted to crop around the knee data more tightly 
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
    

    """
    #OPTION 3: same as above, however it uses a tighter cropping around the knee data, such that the slices are ultimately 362 x362 rather than 424 x424
    #we make a circumcircle around where the knee data is in the image, and padd the image to fit the circumcircle as an incircle
    
    circumcircleRadius = int(math.sqrt(128**2 + 128**2))
    newHeight, newWidth = (circumcircleRadius *2,circumcircleRadius *2)
    newVolume =  np.ndarray((300, newHeight, newWidth))

    topLeft = (circumcircleRadius - 128, circumcircleRadius - 128)
    topRight = (circumcircleRadius - 128, circumcircleRadius + 128)
    bottomLeft = (circumcircleRadius + 128, circumcircleRadius - 128)
    bottomRight = (circumcircleRadius + 128, circumcircleRadius + 128)

    rowOfZeros = np.array([0 for i in range (0, newWidth)])

    
    #add the black background around the circumcircle
    for k in range (0,300):
        image = mask[k]
        image = np.where(image[20:276] !=0,1,0)  
        newImage = np.ndarray((newHeight, newWidth))
        for y in range (0, topLeft[0]):
            newImage[y] = rowOfZeros
        for y in range (topLeft[0], bottomLeft[0]):
            for x in range (0, topLeft[1]):
                newImage[y][x] = 0
            for x in range (topLeft[1], topRight[1]):
                row = image[y-topLeft[0]][11: 139] #this is the row which, once douled, we want to sit within topleft[1] and topright[1]
                i , j= divmod((x- topLeft[1]),2)
                newImage[y][x] = row[i]
            for x in range (topRight[1], newWidth):
                newImage[y][x] = 0
        for y in range (bottomLeft[0], newHeight):
            newImage[y] = rowOfZeros
        newVolume[k] = newImage
    return newVolume
    """



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
def generateTrueCoronalTrainingData(patients, patientAngles, filename, save = True, side = "LEFT", without_cartilidge = False):
    allImages= np.ndarray((len(patients)*300, 424,424))
    for i in range(0,len(patients)):
        print(i)
        patient = patients[i]
        _, mask = loadPatientMask(patient, side, "1")
        if without_cartilidge:
            femoralBone = (cv2.threshold(mask, 1, 7, cv2.THRESH_TOZERO_INV))[1]  #1 in the array
            tibialBone = cv2.threshold((cv2.threshold(mask,1,7,cv2.THRESH_TOZERO))[1], 2,7, cv2.THRESH_TOZERO_INV)[1] #2 in the array
            patella = cv2.threshold((cv2.threshold(mask,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
            boneOnly = np.add(np.add(femoralBone, tibialBone), patella)
            mask = boneOnly
        paddedMask = prepVolumeWithCircumCircleNew(mask)
        for j in range (0, 300):
            allImages[300*i + j] = rotate_incircle(paddedMask[j], patientAngles[i])
    if save:
        currDir = os.curdir
        os.chdir("data")
        np.savez_compressed(filename, x=allImages)
        os.chdir(currDir)
    return allImages

def rotate_volume(mask, angle, alreadyPadded = False, without_cartilidge = False):
    if not alreadyPadded:
        if without_cartilidge:
            femoralBone = (cv2.threshold(mask, 1, 7, cv2.THRESH_TOZERO_INV))[1]  #1 in the array
            tibialBone = cv2.threshold((cv2.threshold(mask,1,7,cv2.THRESH_TOZERO))[1], 2,7, cv2.THRESH_TOZERO_INV)[1] #2 in the array
            patella = cv2.threshold((cv2.threshold(mask,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
            boneOnly = np.add(np.add(femoralBone, tibialBone), patella)
            mask = boneOnly
        paddedMask = prepVolumeWithCircumCircleNew(mask)
    else:
        paddedMask = mask
    newMask = np.ndarray(paddedMask.shape)
    for j in range (0, paddedMask.shape[0]):
        newMask[j] = rotate_incircle(paddedMask[j], angle)
    return newMask




#patientAngles = [350, 353, 349, 353, 352, 358, 352, 350, 359, 357]
#patients = ["9911221", "9911721", "9912946", "9917307", "9918802", "9921811", "9924274", "9937239", "9938236", "9943227"]
#patientAngles = [350, 356, 351]
#patients = ["9947240", "9958234", "9964731"]



#patientAngles = [0, 0]
#patients = ["9947240", "9958234"]
#patients = ["9911221"]
#patientAngles = [350]
#generateTrueCoronalTrainingData(patients, patientAngles, "genuineUnseenOffsets.npy")



#folder, mask = loadPatientMask("9911221", "RIGHT", "1")

#croppedMask = prepVolumeWithCircumCircleNew(mask)
#print(croppedMask.shape)
#pseudo_xray = MRI_to_Xray(croppedMask, view = "coronal")
#plt.imshow(pseudo_xray, cmap ='gray')
#plt.show()
#plt.imshow(croppedMask[95])
#plt.show()
#plt.imshow(rotate_incircle(croppedMask[95], 25), cmap ='gray')
#plt.show()


trainingAngles = [350, 353, 349, 353, 352, 358, 352, 350, 359, 357]
trainingPatients = ["9911221", "9911721", "9912946", "9917307", "9918802", "9921811", "9924274", "9937239", "9938236", "9943227"]
testingAngles = [350, 356, 351, 355, 356, 356, 346, 351, 352, 357]
testingPatients = ["9947240", "9958234", "9964731", "9002116", "9000622","9002316", "9002411", "9002430", "9002817","9003126"]
#9002411 is a BRILLIANT EXAMPLE

"""
i=9
testerView = generateTrueCoronalTrainingData(testingPatients[i:i+1], [0], "tester.npy", save=False)
pseudo_xray = MRI_to_Xray(testerView)
plt.imshow(pseudo_xray, cmap ='gray')
plt.show()
"""


#generateTrueCoronalTrainingData(testingPatients, [0,0,0,0,0,0,0,0,0,0], "unseen_originals.npy")



rightlegTrainPatients = ["9911221", "9911721","9912946","9917307", "9918802"]
rightLegTrainAngles = [15,8, 10,7,8]



"""
rotatedMask = generateTrueCoronalTrainingData(["9918802"],[8], "tester.npy",save = False, side = "RIGHT")
pseudo_xray = MRI_to_Xray(rotatedMask)
plt.imshow(pseudo_xray, cmap ='gray')
plt.show()
"""


#generateTrueCoronalTrainingData(rightlegTrainPatients, rightLegTrainAngles, "right_corrected_5", side = "RIGHT")
#generateTrueCoronalTrainingData(rightlegTrainPatients, [0,0,0,0,0], "right_originals_5", side = "RIGHT")
#generateTrueCoronalTrainingData(trainingPatients, trainingAngles, "left_corrected_10")
#generateTrueCoronalTrainingData(trainingPatients, [0,0,0,0,0,0,0,0,0,0], "left_original_10")

#generateTrueCoronalTrainingData(["9911221"], [350], "9911221_nocart_corrected", save = True, side = "LEFT", without_cartilidge = True)