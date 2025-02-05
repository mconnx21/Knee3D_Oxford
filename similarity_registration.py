import numpy as np
from newRotateForMasks import rotate_volume
from MRI_to_Xray import MRI_to_Xray
from matplotlib import pyplot as plt
import cv2
from readMLMasks import loadPatientMask


"""Newer verson of this is intensity_registration.py and this needs to be cleaned and archived"""


def pre_process_xray(xray_image):
    xray_image = cv2.threshold(xray_image, 100, 255, cv2.THRESH_TOZERO)[1]  #was 114 for 9911221, 100 for 9947240
    brightest_mask = cv2.threshold(xray_image, 147, 255, cv2.THRESH_TOZERO)[1]
    xray_image = 0.7*xray_image + 0.3*brightest_mask
    xray_image = cv2.blur(xray_image, (4,4))
    xray_image = 5/6 * xray_image


    return xray_image

def get_xray_patella(xray_patella_mask, contour_colour = "black"):
    if contour_colour=="black":
        xray_patella_mask = (255 - xray_patella_mask)
    contours, hierarchy = cv2.findContours(xray_patella_mask,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    return xray_patella_mask, contours


def get_pxray_patella(patella_volume):
    #print("here")
    
    #just_patella = cv2.threshold(MRI_to_Xray(patella_volume), 0.01,255,0)[1]
    just_patella = np.where(MRI_to_Xray(patella_volume)>= 0.01,255,0).astype(np.uint8)
    #just_patella = cv2.cvtColor(just_patella, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(just_patella,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # Draw all contours 
    # -1 signifies drawing all contours 
    contour_mask = np.zeros(just_patella.shape)
    cv2.drawContours(contour_mask, contours, -1, 255, 3) 
    #print(contour_mask.shape)
    #contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_RGB2GRAY)
    return contour_mask, contours

def scale(xray, xray_patella_contours, pxray, pxray_patella_contours):
    print("Number of xray contours passed is ", len(xray_patella_contours))
    print("Number of pxray contours passed is ", len(pxray_patella_contours))
    assert (len(xray_patella_contours)==1 and len(pxray_patella_contours)==1)
    
    xray_rect = cv2.boundingRect(xray_patella_contours[0])
    pxray_rect = cv2.boundingRect(pxray_patella_contours[0])
    width_scale_factor = pxray_rect[2] / xray_rect[2]
    height_scale_factor = pxray_rect[3] / xray_rect[3]

    width_scale_factor = max(width_scale_factor, height_scale_factor) #keeping aspect ratio the same
    height_scale_factor = width_scale_factor



    #print(width_scale_factor, height_scale_factor)
    scaled_xray = cv2.resize(xray, (0,0), fx = width_scale_factor, fy = height_scale_factor)
    pxray_centre = (int(pxray_rect[0]+ (pxray_rect[2]//2)), int(pxray_rect[1]+ (pxray_rect[3]//2)))
    xray_centre = (int(xray_rect[0]*width_scale_factor+ (xray_rect[2]*width_scale_factor//2)), int(xray_rect[1]*height_scale_factor+ (xray_rect[3]*height_scale_factor//2)))
    return scaled_xray, pxray_centre, xray_centre

def crop_scaled_xray(scaled_xray, pxray_centre, xray_centre, pxray):
    #print(xray_centre)
    #marked_xray = cv2.circle(scaled_xray, xray_centre, 2, 255, -1)
    #plt.imshow(marked_xray)
    #plt.show()
    #marked_pxray = cv2.circle(pxray, pxray_centre, 2, 255, -1)
    #plt.imshow(marked_pxray)
    #plt.show()
    #print(scaled_xray.shape)
    #print(pxray.shape)


    left_bound = xray_centre[0]-pxray_centre[0]
    right_bound = left_bound + pxray.shape[1]
    top_bound = xray_centre[1]-pxray_centre[1]
    bottom_bound = top_bound+pxray.shape[0]
    #print(top_bound, bottom_bound, left_bound, right_bound)

    cropped_xray = np.transpose(scaled_xray)
    cropped_xray = np.concatenate([cropped_xray, np.array([[0 for i in range(0,cropped_xray.shape[1])] for j in range (0,50)])])
    cropped_xray = np.transpose(cropped_xray)

    cropped_xray = cropped_xray[top_bound:bottom_bound, left_bound:right_bound]
    overlay = 0.5*cropped_xray + 0.5*pxray
    plt.imshow(overlay, cmap = 'gray')
    plt.show()
    """
    cropped_xray = cropped_xray[top_bound:bottom_bound]
    
    #plt.imshow(cropped_xray)
    #plt.show()
    print(cropped_xray.shape)
    cropped_xray = np.transpose(cropped_xray)
    print(cropped_xray.shape)
    cropped_xray = cropped_xray[left_bound:right_bound]
    print(cropped_xray.shape)
    cropped_xray = np.transpose(cropped_xray)
    print(cropped_xray.shape)
    #plt.imshow(cropped_xray)
    #plt.show()
    """
    fg,ax = plt.subplots(1,2)
    ax[0].imshow(cropped_xray, cmap = 'gray', vmin = 0, vmax = 255)
    ax[1].imshow(pxray, cmap= 'gray', vmin = 0, vmax = 255)
    plt.show()

    



    return cropped_xray, top_bound, bottom_bound, left_bound, right_bound

#https://kclpure.kcl.ac.uk/ws/portalfiles/portal/12065027/Studentthesis-Graeme_Penney_2000.pdf
#make sure inputs are numpy arrays and not just lists
def entropy_of_difference(image1, image2, scale,num_bins):
    assert(image1.shape == image2.shape)
    difference_array = image1 - scale*image2

    (minVal, maxVal) = (np.min(difference_array), np.max(difference_array))
    #print(minVal, maxVal)

    bin_width = ((maxVal-minVal)//num_bins)+1

    bin_counts = [0 for i in range (0, num_bins)]

    def find_index_of_bin(value):
        k = 0
        while( minVal + k*bin_width <= value  and k<num_bins):
            k+=1
        
        k-=1
        return k

    
    for i in range (0, difference_array.shape[0]):
        for j in range (0, difference_array.shape[1]):
            value = difference_array[i][j]
            bin = find_index_of_bin(value)
            bin_counts[bin] +=1
    
    total_num_values = difference_array.shape[0]*difference_array.shape[1]
    entropy = 0
    for bin in range(0, num_bins):
        prob = bin_counts[bin]/total_num_values
        if prob != 0:
            entropy += prob*np.log(prob)
    entropy = -1*entropy

    return entropy

def plot_entropies(volume, start, num_images, step, xray, scale, num_bins):
    entropies = []

    for i in range (0, num_images):
        angle = start + step*i
        
        rotated_volume = rotate_volume(volume, angle, alreadyPadded=True)
        pseudo_xray = MRI_to_Xray(rotated_volume)
        
        entropy = entropy_of_difference(xray, pseudo_xray, scale, num_bins)
        entropies.append(entropy)
    
    return entropies

def two_degrees_freedom(xray, xray_patella, volume, start, angle_step, x_step):
    return False

def rotational_freedom(volume, start, num_images, step, xray, xray_patella_mask, scalar, num_bins):
    patella_volume = cv2.threshold((cv2.threshold(volume,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
    xray_patella_mask, xray_contours = get_xray_patella(xray_patella_mask)
    entropies = []

    for i in range (0, num_images):
        angle = (start + step*i)%360

    
        rotated_patella_volume = rotate_volume(patella_volume,angle)
        contour_mask, pxray_contours = get_pxray_patella(rotated_patella_volume)


        
        rotated_volume = rotate_volume(volume, angle, without_cartilidge=True)
        pxray = 255*MRI_to_Xray(rotated_volume)

        
        scaled_xray, pxray_centre, xray_centre = scale(xray, xray_contours[2:3], pxray, pxray_contours)
        cropped_xray ,top_bound, bottom_bound, left_bound, right_bound= crop_scaled_xray(scaled_xray, pxray_centre, xray_centre, pxray)
        print(top_bound, left_bound)

        overlay = 0.5*cropped_xray + 0.5*pxray
        plt.imshow(overlay, cmap ='gray')
        plt.show()
        

        entropy = entropy_of_difference(cropped_xray, pxray, scalar, num_bins)
        entropies.append(entropy)
        print(entropy)
    
    return entropies

#given the original xray, the current position of it, and the pxray, test different vertical translations
def vertical_translational_freedom(xray, top_bound, left_bound, num_up, num_down, step, pxray, scalar, num_bins):
    entropies = []

    
    i = 0
    top = top_bound - num_up*step
    h, w = pxray.shape
    while (top <0):
        top += step
        num_up -=1

    heights = []
    
    total_num_trans = num_up + num_down
    while (i< total_num_trans):
        print("Top postition is ", top)
        heights.append(top)
        #print(w, h)
        #print(xray.shape)
        cropped_xray = xray[top:(top+h), left_bound:(left_bound+w)] 
        #print(cropped_xray.shape)

        overlay = 0.5*cropped_xray + 0.5*pxray
        plt.imshow(overlay, cmap ='gray')
        plt.show()
        


        entropy = entropy_of_difference(cropped_xray, pxray, scalar, num_bins)

        entropies.append(entropy)
        print("Entropy of  this angle at height ", top, "is ", entropy)


        top += step
        i+=1
    return entropies, heights
    

def rotation_and_vert(volume, rot_start, rot_num_images, rot_step, xray, xray_patella_mask, scalar, num_bins, num_up, num_down, trans_step):
    patella_volume = cv2.threshold((cv2.threshold(volume,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
    xray_patella_mask, xray_contours = get_xray_patella(xray_patella_mask)
    #contoured_xray = cv2.drawContours(xray, xray_contours, -1, 255, 1)
    #plt.imshow(contoured_xray)
    #plt.show()
    #xray_contour = xray_contours[2:3] #for 9911221 , need to fix
    print(len(xray_contours))
    #xray_contour = xray_contours[1:2] # for 9947240 need to fix
    entropies = []
    angles = []

    for i in range (0, rot_num_images):
        
        angle = (rot_start + rot_step*i)%360
        angles.append(angle)
        print("Angle ", angle)
        print("------------------------")

    
        rotated_patella_volume = rotate_volume(patella_volume,angle)
        contour_mask, pxray_contours = get_pxray_patella(rotated_patella_volume)


        
        rotated_volume = rotate_volume(volume, angle, without_cartilidge=True)
        pxray = 255*1.3*MRI_to_Xray(rotated_volume)

        
        scaled_xray, pxray_centre, xray_centre = scale(xray, xray_contours, pxray, pxray_contours)
        cropped_xray ,top_bound, bottom_bound, left_bound, right_bound = crop_scaled_xray(scaled_xray, pxray_centre, xray_centre, pxray)

        print(top_bound, left_bound)
        print("Starting vertical translation entropies for angle ", str(angle))

        vert_entropies, heights = vertical_translational_freedom(scaled_xray, top_bound, left_bound, num_up, num_down, trans_step, pxray, scalar, num_bins)

        

        #overlay = 0.5*cropped_xray + 0.5*pxray
        #plt.imshow(overlay, cmap ='gray')
        #plt.show()
        

        entropy = np.min (vert_entropies)
        height_of_min_entropy = heights[np.argmin(vert_entropies)]
        entropies.append((entropy, height_of_min_entropy))
        print("Minimum entropy for angle ", str(angle), " is ", entropy, " at height ", height_of_min_entropy)
    print(entropies)
    print(min(entropies))
    min_entropy, height_achieved_at = min(entropies)
    angle_achieved_at = angles[np.argmin(entropies)]
    
    return entropies, min_entropy, angle_achieved_at, height_achieved_at



"""
volume = np.load("data\\9911221_nocart_corrected.npz")['x']
fg, ax = plt.subplots(1,4)
#xray = MRI_to_Xray(volume)
xray = cv2.imread("xrays\\enhanced_practice_xray.png")
xray = cv2.cvtColor(xray, cv2.COLOR_RGB2GRAY)
xray = xray[33:633]
xray = np.transpose(xray)
xray = xray[48: (945-49)]
xray = np.transpose(xray)


xray = cv2.resize(xray, (0,0), fx= 0.5, fy=0.5)
xray = (cv2.threshold(xray,40,255,cv2.THRESH_TOZERO))[1]
xray = cv2.blur(xray, (8,8))
zero_rotate = MRI_to_Xray(rotate_volume(volume, 9, alreadyPadded=True))

print(xray.shape)
print(zero_rotate.shape)
print(xray)
print(zero_rotate)
print(np.max(zero_rotate))
diff = xray- 255*zero_rotate
overlay = 0.3*xray + 0.7*255*zero_rotate
ax[0].imshow(overlay, cmap = 'gray')
ax[1].imshow(diff, cmap= 'gray')
ax[2].imshow(xray, cmap = 'gray')
ax[3].imshow(255*zero_rotate, cmap = 'gray')
#print(diff)
plt.show()


start = 0
num_images = 15
step =2
scale = 255
num_bins = 64
print(plot_entropies(volume, start, num_images, step, xray, scale, num_bins))



"""
"""
_, volume = loadPatientMask("9911221", "LEFT", "1")
patella_volume = cv2.threshold((cv2.threshold(volume,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
rotated_patella_volume = rotate_volume(patella_volume,3)
#rotated_patella_volume = np.load("data\\rotated_patella_volume.npz", allow_pickle=True)['x']
#np.savez_compressed("data\\rotated_patella_volume.npz", x= rotated_patella_volume)
contour_mask, pxray_contours = get_pxray_patella(rotated_patella_volume)
rotated_volume = rotate_volume(volume,3, without_cartilidge=True)
#rotated_volume = np.load("data\\rotated_volume.npz", allow_pickle=True)['x']
#np.savez_compressed("data\\rotated_volume.npz", x= rotated_volume)
pxray = 255*MRI_to_Xray(rotated_volume)

#image = contour_mask + 255*pxray
#cv2.drawContours(pxray, contours, -1, 255, 3) 
#plt.imshow(pxray, cmap ='gray')
#plt.show()


xray_patella_mask = cv2.imread("xrays\\9911221_left_xray_contour.png", cv2.IMREAD_GRAYSCALE)
#plt.imshow(xray_patella_mask)
#plt.show()
xray_patella_mask, xray_contours = get_xray_patella(xray_patella_mask)
print(len(xray_contours), len(pxray_contours))
xray = pre_process_xray(cv2.imread("xrays\\left_xray.jpg", cv2.IMREAD_GRAYSCALE))
#contoured_xray = cv2.drawContours(xray, xray_contours[2:3],-1, 255,3)  #need to fix the fact that im getting multiple xray conoturs but roll with it for now
#contoured_pxray = cv2.drawContours(pxray, pxray_contours, -1, 255, 3)
#plt.imshow(contoured_xray)
#plt.show()
#plt.imshow(contoured_pxray)
#plt.show()
cropped_xray , _,_,_,_= scale(xray, xray_contours[2:3], pxray, pxray_contours)
print(cropped_xray.shape)

overlay = 0.5*cropped_xray + 0.5*pxray
plt.imshow(overlay, cmap = 'gray')
plt.show()

print(entropy_of_difference(cropped_xray, pxray, 1, 64))
"""
patient = "9947240"
_, volume = loadPatientMask(patient, "LEFT", "1")

xray_patella_mask = cv2.imread("xrays\\"+patient+"_left_xray_contour_new.png", cv2.IMREAD_GRAYSCALE)
xray = pre_process_xray(cv2.imread("xrays\\"+patient+"_left_xray.jpg", cv2.IMREAD_GRAYSCALE))
#rotational_freedom(volume, 0, 2, 1, xray, xray_patella_mask, 1, 64)
#xray_patella_mask, xray_contours = get_xray_patella(xray_patella_mask)
#print(len(xray_contours))
#contoured_xray = cv2.drawContours(xray, xray_contours[1:2],-1, 255,3)  #needed 2:3 for 9911221
#plt.imshow(contoured_xray, cmap = 'gray')
#plt.show()


entropies, min_entropy, angle_achieved_at, height_achieved_at = rotation_and_vert(volume, 344, 2,4,xray, xray_patella_mask, 1, 64, 0,5,4)
print(entropies)
print("Min entropy was ", min_entropy, " achieved at (angle, height) ", angle_achieved_at, height_achieved_at)











