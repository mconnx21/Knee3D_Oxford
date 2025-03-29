import math
import numpy as np
import cv2
import os
from pathlib import Path
from matplotlib import pyplot as plt
import random
import SimpleITK as sitk
import skimage
from skimage import measure
import sys
import pydicom


f = open("requirements.txt", "w")
f.write("Python " + sys.version + " \n")
f.write("math "+ " \n")
f.write("numpy "+str(np.__version__)+ " \n")
f.write("cv2 "+str(cv2.__version__)+ " \n")
f.write("os "+ " \n")
f.write("pathlib.Path "+" \n")
f.write("matplotlib.pyplot "+ " \n")
f.write("random "+ " \n")
f.write("SimpleITK "+str(sitk.__version__)+ " \n")
f.write("skimage.measure "+str(skimage.__version__)+ " \n")
f.write("sys "+ " \n")
f.write("pydicom "+str(pydicom.__version__)+ " \n")
f.close()