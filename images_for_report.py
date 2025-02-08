from newRotateForMasks import *
from MRI_to_Xray import MRI_to_Xray
from matplotlib import pyplot as plt

from readMLMasks import loadPatientMask
import cv2
from intensity_registration import pre_process_xray_tibialvals, pre_process_pxray
import os


def old_rotate(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def coronal_axial_correspondance(patient, side, angle, axial_slice_num):
    #pseudo_mri_axial_correspondance
    _, mask = loadPatientMask(patient, side, "1")

    prepped_mask = prepVolumeWithCircumCircleNew(mask)
    original_image = prepped_mask[axial_slice_num]
    original_pseudoxray = MRI_to_Xray(prepped_mask)

    true_coronal_mask = rotate_volume(prepped_mask, angle, alreadyPadded=True)
    true_coronal_image = true_coronal_mask[axial_slice_num]
    true_coronal_pseudoxray = MRI_to_Xray(true_coronal_mask)

    fig, ax = plt.subplots(1,4)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax[0].imshow(original_pseudoxray, cmap ='gray')
    ax[0].set_title("10°, Coronal View")
    ax[0].axis('off')
    ax[1].imshow(original_image, cmap= 'gray')
    ax[1].set_title("10°, "+ str(axial_slice_num)+"th Axial Slice")
    ax[1].axis('off')
    ax[2].imshow(true_coronal_pseudoxray, cmap ='gray')
    ax[2].set_title("0°, Coronal View")
    ax[2].axis('off')
    ax[3].imshow(true_coronal_image, cmap= 'gray')
    ax[3].set_title("0°, "+ str(axial_slice_num)+"th Axial Slice")
    ax[3].axis('off')
    plt.show()

    #old_rotate_issue
    oldoriginal = original_image
    original_image = np.ndarray((oldoriginal.shape[0], oldoriginal.shape[1], 3))
    old_truecoronal = true_coronal_image
    true_coronal_image = np.ndarray((oldoriginal.shape[0], oldoriginal.shape[1], 3))
    for i in range (oldoriginal.shape[0]):
        for j in range (oldoriginal.shape[1]):
            if oldoriginal[i][j] >0:
                original_image[i][j] = (1,1,1)
            else:
                original_image[i][j] = (0,0,0)
            if old_truecoronal[i][j] > 0:
                true_coronal_image[i][j] =(1,1,1)
            else:
                true_coronal_image[i][j] = (0,0,0)
    #original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    borderedImage = cv2.copyMakeBorder(src=original_image, top=15, bottom=15, left=15, right=15, borderType=cv2.BORDER_CONSTANT, value = (1,0,0))
    oldRotatedImage = old_rotate(borderedImage, 350)
    circledImage = cv2.circle(original_image, (original_image.shape[0]//2,original_image.shape[1]//2), original_image.shape[0]//2, (1,0,0), 15)
    circledRotatedImage= cv2.circle(true_coronal_image, (original_image.shape[0]//2,original_image.shape[1]//2), original_image.shape[0]//2, (1,0,0), 15)
    fig, ax = plt.subplots(1,4)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax[0].imshow(borderedImage)
    ax[0].set_title("10°, Original")
    ax[0].axis('off')
    ax[1].imshow(oldRotatedImage,)
    ax[1].axis('off')
    ax[1].set_title("0°, Old Rotation Function")
    ax[2].imshow(circledImage)
    ax[2].set_title("10°, Original")
    ax[2].axis('off')
    ax[3].imshow(circledRotatedImage)
    ax[3].axis('off')
    ax[3].set_title("0°, New Rotation Function")
    plt.show()

#coronal_axial_correspondance("9911221", "LEFT", 350, 100)



def xray_processing_image(patient, side, xray_patella_centre, tibial_coordinates, kernel_size):
    original_xray = cv2.imread("xrays\\"+patient+"_"+side.lower()+"_xray.jpg", cv2.IMREAD_GRAYSCALE)
    xray_patella_mask = cv2.imread("xrays\\"+patient+"_"+side.lower()+"_xray_contour.png", cv2.IMREAD_GRAYSCALE)
    path = "report\\xray_processing_image\\"+patient+"\\"+ side+"\\" 
    if not os.path.exists(path):
        os.makedirs(path)
    

    xray = pre_process_xray_tibialvals(original_xray, xray_patella_mask, xray_patella_centre, tibial_coordinates, kernel_size, save = True, save_location=path)


"""

i = patients.index("9031961")
xray_processing_image("9031961", "LEFT", patient_pat_centres[i], patient_tib_centres[i], 32)
"""

def get_pxray(patient, side, angle):
    resize = 1
    _, volume = loadPatientMask(patient, side, "1")
    patella_volume = cv2.threshold((cv2.threshold(volume,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
    rotated_patella_volume = rotate_volume(patella_volume,angle)
    just_patella = cv2.resize(np.where(MRI_to_Xray(rotated_patella_volume)>= 0.01,255,0).astype(np.uint8),(0,0), fx=resize, fy=resize)
    rotated_volume = rotate_volume(volume, angle, without_cartilidge=True)
    pxray = pre_process_pxray(rotated_volume, just_patella, 85, 115, resize)

    path = "report\\pxrays\\"+patient+"\\"+ side+"\\"
    if not os.path.exists(path):
        os.makedirs(path)
    path = path  +"angle"+str(angle) +"_pxray.png"
    
    cv2.imwrite(path, pxray)



patients = ["9002316", "9002411", "9002817", "9911221", "9911721", "9917307", "9918802", "9921811", "9924274", "9947240", "9938236", "9943227", "9958234", "9964731", "9986355", "9986838", "9989352", "9989700", "9990192", "9990355", "9986207","9030925", "9031141", "9031930", "9031961", "9033937", "9034451", "9034677", "9034812", "9034963"]
patient_tib_centres = [(170,300), (170,300),(170, 270), (170,303),(170,239), (185,325), (121,286), (179,321),(207,312) , (170,339), (181, 317), (203,326), (155,343),(183,269), (200,310), (158,307), (192,298), (170,300), (170,311), (179,338), (212,311),(192,335), (150,266), (180,354), (164,332), (178,350), (145,296), (156,256), (159,317), (209,329)]
patient_pat_centres = [(170,220), (170,196), (170,186), (155,232),(182,169), (170,200), (103,195), (178,226), (199, 208), (169,238), (174, 236), (206,240), (146,239), (209,172), (211,216), (153,198), (193,208), (178,206), (164,222), (170,201), (197,222), (204,158), (138,160), (215,195), (156,235), (170,197), (148,199), (151,167), (153,232), (209,238)]

i = patients.index("9964731")
xray_processing_image(patients[i], "LEFT", patient_pat_centres[i], patient_tib_centres[i], 32)
get_pxray(patients[i], "LEFT", 0)
