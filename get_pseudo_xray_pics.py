from MRI_to_Xray import MRI_to_Xray
from newRotateForMasks import *
from readMLMasks import loadPatientMask


_, mask = loadPatientMask("9911221", "LEFT", "1")
rotated_volume = rotate_volume(mask, 0, without_cartilidge=True)

coronal =  MRI_to_Xray(rotated_volume, "coronal")
sagittal =  MRI_to_Xray(rotated_volume, "saggital")
axial =  MRI_to_Xray(rotated_volume, "axial")
plt.imshow(sagittal, cmap="gray")
plt.axis("off")
plt.savefig("sagittal_pxray_9911221.png", bbox_inches ="tight")