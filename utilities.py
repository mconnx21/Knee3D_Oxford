import pydicom
import os
import numpy as np
from matplotlib import pyplot as plt


#INPUT: a path to a dicom folder containing a series of dicom slices
#OUTPUT: a 3D array containing the slices (note this is not scaled to within 0 and 255 and is the absolute intensity value), the distance betwwen slices in mm,
# the real world distance in mm correspinding to the pixel width in one slice, and the dimensions of the 3D array
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


#INPUT: a 3D array of the pure dicom data volume, and the number of copies of each slice needed to "fill the slice gap" i.e. make the 3D volume in proportion
#OUTPUT: a 3D numpy array of the volume with gaps filled
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

#INPUT: a 3D array with dimensions (x,y,z)
#INPUT: A 3D array with dimensions (x, y, 2*z)
#note that this isn't the same as fillGaps - we are not duplicating slcies, we are "stretching" each slice in z dimension
# the motivation for this function is the axial slices given by the ML segments, which require stretching to be proportionate
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

#INPUT: a 2D array representing a single slice image of a DICOM
#OUTPUT: no return value, but an image corresponding to the DICOM slice will be displayed.
#NB:  this is a crude implementation which doesn't appropriately scale the intensitiy valyues in the slice to be between 0 and 255
#however it will still display enough information for quick testing purposes.
def displayASlice(slice):
    fig, ax = plt.subplots()
    fig.set_facecolor('black')
    ax.imshow(slice, cmap ='gray', vmin = 0, vmax = 255)
    plt.show()


#INPUT: the filename e.g. tibialBone.ply, the points and the normals of the point cloud you want to save. "points" and "normals" are given as arrays, as outputted by the marching cubes algorithm
# The function will save the point cloud in ply format
#
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