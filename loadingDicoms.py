import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt


"""This is old code which needs to be cleaned up  / removed.
It initially allows you to read dicom files, and then in theorey view them from either axial, saggital, or coronal,
However there are better ways to do this now."""


pathToDicom = "oai\\Package_1216539_samples\\1.C.2\\9999865\\20060331\\11077102"

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
print(pixelDistanceSaggital)
print(sliceThicknessSaggital / distanceBetweenZSaggital)


#print(len(DCMFiles))
DCMArrays = np.array(DCMArrays)
"""
print(DCMArrays[0])
print(len(DCMArrays))
print(DCMArrays[75])
print(DCMArrays.shape)
"""

"""next steps are to define our axis"""
# 
#Suppose we are looking ath the saggital view
# let the origin be the top left corner of the screen
# +x comes out of the screen, +y goes down the screen vertically, +z goes accross the screen left to right

#DEEPER EXPALANTION:
# Since our slices are given by DCMArrays[i], we will consider the x axis to be 0 on the screen and +x gives slices coming out of the screen
# DCM[0][y] gives us the first slice, and the first row of pixels from top to bottom
# So +y goes down the screen from the top left corner



""" then do 90 degree reconstructions and compare to the ones radiant produces view MPR"""
#we now want to imagine that the viewer has a gaze g paralell to positive z, and we are orthographically projecting onto the plane z = 0
#but actually we are simply changing the shape of the array, that's all
#we are taking, for each z in 0 to 383, take the zth column of each saggital slice in the x axis, and concatenate them together 

def saggitalToCoronal(arr, sliceThicknessSaggital, distanceBetweenZSaggital):
    newSliceThickness = int(round(sliceThicknessSaggital / distanceBetweenZSaggital, 0))

    #we want a total of newSliceThickness copies of each new slice that we generate, so that the distance between pixels in the new slices we generate is approximately the same scale as the original images


    depth, height, width = arr.shape
    coronalArray = np.ndarray((width, height, depth*newSliceThickness), int)
    
    for z in range(0, width):
        #take plane Z = z info, which is the zth columns of the saggital slices, all concatted together into a matrix
        thisNewSlice = []
        for x in range(0, depth):
            #find zth column of this x slice 
            zthColumnThisSlice = np.transpose(arr[x])[z]
            #going to add all the columns as rows then transpose at the end
            for i in range (0, newSliceThickness):
                thisNewSlice.append(zthColumnThisSlice)
        thisNewSlice = np.transpose(np.array(thisNewSlice)) # thisNewSlice[i] is a row of pixels paralell to +x in old system, thisNewSlice columns are paralell to +y in old corrdiante system
        coronalArray[z] = thisNewSlice
        
    
    #coronalArray[z] is a slice at Z=z in old axis.
    #so now iterating through the slices gives orthogonal projection for a viewer looking down positive z gaze
    return coronalArray


def saggitalToAxial(arr, sliceThicknessSaggital, distanceBetweenYSaggital):
    newSliceThickness = int(round(sliceThicknessSaggital / distanceBetweenYSaggital, 0))

    #we want a total of newSliceThickness copies of each new slice that we generate, so that the distance between pixels in the new slices we generate is approximately the same scale as the original images

    depth, height, width = arr.shape
    axialArray = np.ndarray((height, width, depth*newSliceThickness), int)
    
    for y in range(0, height):
        #take plane Y = y info, which is the zth columns of the saggital slices, all concatted together into a matrix
        thisNewSlice = []
        for x in range(0, depth):
            #find yth row of this x slice 
            ythRowThisSlice = arr[x][y]
            #going to add all the columns as rows then transpose at the end
            for i in range (0, newSliceThickness):
                thisNewSlice.append(ythRowThisSlice)
        thisNewSlice = np.transpose(np.array(thisNewSlice)) # thisNewSlice[i] is a row of pixels paralell to +x in old system, thisNewSlice columns are paralell to +y in old corrdiante system
        axialArray[y] = thisNewSlice
        
    
    #coronalArray[z] is a slice at Z=z in old axis.
    #so now iterating through the slices gives orthogonal projection for a viewer looking down positive z gaze
    return axialArray

coronal = saggitalToCoronal(DCMArrays, sliceThicknessSaggital, distanceBetweenZSaggital)
axial = saggitalToAxial(DCMArrays, sliceThicknessSaggital, distanceBetweenYSaggital)
#print(coronal[75])

fig, ax = plt.subplots()
fig.set_facecolor('black')
ax.imshow(axial[75], cmap ='gray', vmin = 0, vmax = 255)
plt.show()











# then work out how to "guess" pixel intensitis where our slice planes aren't on a known value
