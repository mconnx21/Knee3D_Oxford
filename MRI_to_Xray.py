
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from pathlib import Path

"""PURPOSE OF THIS SCRIPT:
Implement functions which takes one of the ML masks and allows you to project in any of saggital, coronal, or axial views
to get an xray-like 2D image.
"""

def MRI_to_Xray(mask, view = "coronal"):
    DCMImage = sitk.GetImageFromArray(mask) #convert to a sci-kit image
    thisFilter = sitk.MeanProjectionImageFilter()
    if view == "axial":

        projectionDirection = 2
    elif view == "saggital":
        projectionDirection = 0
    else:
        projectionDirection = 1

    thisFilter.SetProjectionDimension(projectionDirection)
    meanProjectedSagg = thisFilter.Execute(DCMImage)
    filteredArray = sitk.GetArrayFromImage(meanProjectedSagg)

    if projectionDirection == 2:
        filteredArray = filteredArray[0]
    if projectionDirection ==1:
        x,_,y = filteredArray.shape
        newArray = np.ndarray((x,y))
        for i in range (0, x):
            for j in range (0,y):
                newArray[i][j] = filteredArray[i][0][j]
        filteredArray = newArray
    return filteredArray



patient1 = "oai\\Package_1216539_samples\\1.C.2\\9911221\\20050909\\10583703"
patient2 = "oai\\Package_1216539_samples\\1.C.2\\9911721\\20050812\\10478403"
patient3 = "oai\\Package_1216539_samples\\1.C.2\\9947240\\20051013\\10200003"
patients = ["9911221", "9911721", "9912946", "9917307", "9918802", "9921811", "9924274", "9937239", "9938236", "9943227"]

"""

#by trial and error manually! In degrees
#I found the rotation angles of the volumes which corresponded to a true coronal pseudo-xray
patientAngles = [350, 353, 349, 353, 352, 358, 352, 350, 359, 357]
patientAnglesUnseen = [350, 356, 351] #third patient here is  particularly bad example
"""

"""
dataFile = "genuineUnseenOffsets.npy"
path = Path("data\\"+dataFile)  #save directory it will be better to later extend this to make subfolders based on angles
tester = np.load(path)
testerMask = tester[0:300]
pseudo_xray = MRI_to_Xray(testerMask, view = "axial")
plt.imshow(pseudo_xray, cmap ='gray')
plt.show()
"""