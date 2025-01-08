import numpy as np
from skimage import measure
from mhdMasks import write_ply_withNormals, pad
from matplotlib import pyplot as plt
import sys
import cv2
from pathlib import Path
import os

"""This file reads segmentations given as np\ files, and allows you to save different PLY for different
tissues. These can then be viewed as 3D meshes. """
np.set_printoptions(threshold = sys.maxsize)

def loadPatientMask(patient,side,visit):
    folder = "MLSegments\\" + patient 
    f = np.load(folder+ "\\" + patient + "_" + side+"_"+visit+".npz")
    maskArray = f['x']
    return folder, maskArray


def savePatientPLYs(patient,side,visit):
    originalDirectory = os.getcwd()
    folder, maskArray = loadPatientMask(patient, side, visit)


    femoralBone = (cv2.threshold(maskArray, 1, 7, cv2.THRESH_TOZERO_INV))[1]  #1 in the array
    tibialBone = cv2.threshold((cv2.threshold(maskArray,1,7,cv2.THRESH_TOZERO))[1], 2,7, cv2.THRESH_TOZERO_INV)[1] #2 in the array
    patella = cv2.threshold((cv2.threshold(maskArray,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
    femoralCart= cv2.threshold((cv2.threshold(maskArray,3,7,cv2.THRESH_TOZERO))[1], 4,7, cv2.THRESH_TOZERO_INV)[1] #4 in the array
    tibialCart= cv2.threshold((cv2.threshold(maskArray,4,7,cv2.THRESH_TOZERO))[1], 5,7, cv2.THRESH_TOZERO_INV)[1] #5 in the array
    patellaCart = cv2.threshold((cv2.threshold(maskArray,5,7,cv2.THRESH_TOZERO))[1], 6,7, cv2.THRESH_TOZERO_INV)[1] #6 in the array
    miniscus = (cv2.threshold(maskArray, 6, 7, cv2.THRESH_TOZERO))[1] #7 in the array

    parts = [femoralBone, tibialBone, patella, femoralCart, tibialCart, patellaCart, miniscus]
    names = ["femoralBone", "tibialBone", "patella", "femoralCart", "tibialCart", "patellaCart", "miniscus"]


    os.chdir(folder)

    for i in range(0, len(parts)):
        part = parts[i]
        name = names[i]
        verts, faces, normals, values = measure.marching_cubes(pad(part), 0) 

        write_ply_withNormals(patient+"_"+side+"_" + name+".ply" , verts, normals)

    os.chdir(originalDirectory)


patient = "9947240"
side = "LEFT"
visit = "1"
#savePatientPLYs(patient, side,visit)