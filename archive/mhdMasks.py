import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
from skimage import measure
import sys
import cv2

"""This file takes masks with an mhd extension, oppens them pads them, and writes them to a ply file 
so the mesh can be viewed in 3D"""


np.set_printoptions(threshold= sys.maxsize)
maskArray = sitk.GetArrayFromImage(sitk.ReadImage("masks\\9091131\\9091131.segmentation_masks.mhd", sitk.sitkFloat32))
(x,y,z) = maskArray.shape

#print(maskArray[200][160])

#tester = (cv2.threshold(maskArray[200], 1, 4, cv2.THRESH_TOZERO_INV))[1]
#print(tester[160])


fibialBone = (cv2.threshold(maskArray, 1, 4, cv2.THRESH_TOZERO_INV))[1]
fibialCartilage = cv2.threshold((cv2.threshold(maskArray,1,4,cv2.THRESH_TOZERO))[1], 2,4, cv2.THRESH_TOZERO_INV)[1]
tibialBone = cv2.threshold((cv2.threshold(maskArray,2,4,cv2.THRESH_TOZERO))[1], 3,4, cv2.THRESH_TOZERO_INV)[1]
tibialCartilage = (cv2.threshold(maskArray, 3, 4, cv2.THRESH_TOZERO))[1]


#print(maskArray[200][160])


def pad(arr):
    (x,y,z) = arr.shape
    paddedMask = np.ndarray((x,y,z*2), bool)


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

    

def write_ply_withNormals(filename, points, normals): #add faces if you want to 
    
    #Write a point cloud to a PLY file.

    #Parameters:
    #- filename: str, the name of the output PLY file.
    #- points: numpy array of shape (N, 3), where N is the number of points.
    
    with open(filename, 'w') as file:
        # Write the PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property float nx\n")
        file.write("property float ny\n")
        file.write("property float nz\n")
        #file.write(f"element face {faces.shape[0]}\n")   -- caused meshlab to crash as too many faces
        #file.write("property list uchar int vertex_indices\n")
        file.write("end_header\n")

        # Write the points
        for i in range (0, len(points)):
            point = points[i]
            normal = normals[i]
            file.write(f"{point[0]} {point[1]} {point[2]} {normal[0]} {normal[1]} {normal[2]}\n")
        #for face in faces:
        #    file.write(f"{face[0]} {face[1]} {face[2]}\n")

"""
tibialBoneVol = pad(tibialBone)
tibialCartVol = pad(tibialCartilage)
fibialBoneVol = pad(fibialBone)
fibialCartVol = pad(fibialCartilage)

TBverts, TBfaces, TBnormals, TBvalues = measure.marching_cubes(tibialBoneVol, 0)
TCverts, TCfaces, TCnormals, TCvalues = measure.marching_cubes(tibialCartVol, 0)
FBverts, FBfaces, FBnormals, FBvalues = measure.marching_cubes(fibialBoneVol, 0)
FCverts, FCfaces, FCnormals, FCvalues = measure.marching_cubes(fibialCartVol, 0)
#print(TBfaces)
write_ply_withNormals("tibialBone.ply", TBverts, TBnormals)#, TBfaces)
write_ply_withNormals("fibialBone.ply", FBverts, FBnormals)#, FBfaces)
write_ply_withNormals("tibialCart.ply", TCverts, TCnormals)#, TCfaces)
write_ply_withNormals("fibialCart.ply", FCverts, FCnormals)#, FCfaces)


"""

