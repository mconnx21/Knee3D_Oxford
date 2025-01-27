from newRotateForMasks import *
from matplotlib import pyplot as plt

from readMLMasks import loadPatientMask
import cv2


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

coronal_axial_correspondance("9911221", "LEFT", 350, 100)


