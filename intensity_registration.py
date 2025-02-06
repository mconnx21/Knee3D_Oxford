import numpy as np
from newRotateForMasks import rotate_volume
from MRI_to_Xray import MRI_to_Xray
from matplotlib import pyplot as plt
import cv2
from readMLMasks import loadPatientMask


def pre_process_xray(xray_image):
    xray_image = cv2.threshold(xray_image, 100, 255, cv2.THRESH_TOZERO)[1]  #was 114 for 9911221, 100 for 9947240
    brightest_mask = cv2.threshold(xray_image, 120, 255, cv2.THRESH_TOZERO)[1] #was 147 for 9911221
    xray_image = 0.7*xray_image + 0.3*brightest_mask
    xray_image = cv2.blur(xray_image, (4,4))
    xray_image = 5/6 * xray_image


    return xray_image
"""
def pre_process_xray_tibialvals(xray_image, xray_patella_mask, xray_patella_centre, tibial_coordinates, kernel_size):
    sample = xray_image[tibial_coordinates[1]-(kernel_size//2): tibial_coordinates[1]+(kernel_size//2), tibial_coordinates[0]-(kernel_size//2):tibial_coordinates[0]+(kernel_size//2)]
    #xray_image[tibial_coordinates[1]-(kernel_size//2): tibial_coordinates[1]+(kernel_size//2), tibial_coordinates[0]-(kernel_size//2):tibial_coordinates[0]+(kernel_size//2)] = 1000*sampl
    #xray_image = cv2.rectangle(xray_image, (tibial_coordinates[1]-(kernel_size//2), tibial_coordinates[0]-(kernel_size//2)), ( tibial_coordinates[1]+(kernel_size//2), tibial_coordinates[0]+(kernel_size//2)), 255)
    #plt.imshow(xray_image, cmap = 'gray')
    #plt.show()
    min_val = np.min(sample)
    xray_image = cv2.threshold(xray_image, min_val, 255, cv2.THRESH_TOZERO)[1]
    PXRAY_TIBIAL_VAL  = 85 # contant
    xray_image = (PXRAY_TIBIAL_VAL / min_val) *xray_image
    #now want to find the average patella brightness so that I can enhance.
    brightest_mask = np.where(xray_patella_mask >0, 0, 1)
    #plt.imshow(brightest_mask,cmap = 'gray')
    #plt.show()
    patella_sample = xray_image[xray_patella_centre[1]-(kernel_size//2): xray_patella_centre[1]+(kernel_size//2), xray_patella_centre[0]-(kernel_size//2):xray_patella_centre[0]+(kernel_size//2)]
    avg_patella_val = np.average(patella_sample)
    #brightest_mask = cv2.threshold(xray_image, avg_patella_val, 255, cv2.THRESH_TOZERO)[1]
    #brightest_mask = cv2.threshold(brightest_mask,0,1,cv2.THRESH_BINARY )[1]

    PXRAY_PAT_AVG = 115
    amount_brightness_to_add = PXRAY_PAT_AVG- avg_patella_val
    xray_image = xray_image + amount_brightness_to_add*brightest_mask
    #xray_image = xray_image + PXRAY_PAT_AVG*brightest_mask
    xray_image = cv2.blur(xray_image, (4,4))
    plt.imshow(xray_image, cmap = 'gray')
    plt.show()
    
    return xray_image
"""

def pre_process_xray_tibialvals(xray_image, xray_patella_mask, xray_patella_centre, tibial_coordinates, kernel_size):
    
    #xray_image[tibial_coordinates[1]-(kernel_size//2): tibial_coordinates[1]+(kernel_size//2), tibial_coordinates[0]-(kernel_size//2):tibial_coordinates[0]+(kernel_size//2)] = 1000*sampl
    #xray_image = cv2.rectangle(xray_image, (tibial_coordinates[1]-(kernel_size//2), tibial_coordinates[0]-(kernel_size//2)), ( tibial_coordinates[1]+(kernel_size//2), tibial_coordinates[0]+(kernel_size//2)), 255)
    #plt.imshow(xray_image, cmap = 'gray')
    #plt.show()

    PXRAY_TIBIAL_VAL  = 80 # contant

    blurred = np.copy(xray_image)
    blurred = cv2.blur(blurred, (11,11))  #more blur for use in adaptive thresholding
    xray_image = cv2.blur(xray_image, (4,4))
    

    thresholded_mask = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)
    
    #plt.imshow(thresholded_mask, cmap = 'gray')
    #plt.show()

    
    
    sample = xray_image[tibial_coordinates[1]-(kernel_size//2): tibial_coordinates[1]+(kernel_size//2), tibial_coordinates[0]-(kernel_size//2):tibial_coordinates[0]+(kernel_size//2)]
    min_val = np.min(sample)
    
    #xray_image = cv2.threshold(xray_image, min_val, 255, cv2.THRESH_TOZERO)[1]
    xray_image = xray_image + (15/255)*thresholded_mask
    
    
    #plt.imshow(xray_image, cmap ='gray')
    #plt.show()
    xray_image = cv2.threshold(xray_image, min_val, 255, cv2.THRESH_TOZERO)[1]
    xray_image = (PXRAY_TIBIAL_VAL / min_val) *xray_image
    
    
    #now want to find the average patella brightness so that I can enhance.
    brightest_mask = np.where(xray_patella_mask >0, 0, 1)
    #plt.imshow(brightest_mask,cmap = 'gray')
    #plt.show()
    patella_sample = xray_image[xray_patella_centre[1]-(kernel_size//2): xray_patella_centre[1]+(kernel_size//2), xray_patella_centre[0]-(kernel_size//2):xray_patella_centre[0]+(kernel_size//2)]
    avg_patella_val = np.average(patella_sample)
    #brightest_mask = cv2.threshold(xray_image, avg_patella_val, 255, cv2.THRESH_TOZERO)[1]
    #brightest_mask = cv2.threshold(brightest_mask,0,1,cv2.THRESH_BINARY )[1]

    PXRAY_PAT_AVG = 115
    amount_brightness_to_add = PXRAY_PAT_AVG- avg_patella_val
    xray_image = xray_image + amount_brightness_to_add*brightest_mask
    #xray_image = xray_image + PXRAY_PAT_AVG*brightest_mask
    
    #plt.imshow(xray_image, cmap = 'gray')
    #plt.show()
    
    return xray_image






def get_xray_patella(xray_patella_mask, contour_colour = "black"):
    if contour_colour=="black":
        xray_patella_mask = (255 - xray_patella_mask)
    contours, hierarchy = cv2.findContours(xray_patella_mask,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    contours =[max(contours, key = cv2.contourArea)]  #validation
    return xray_patella_mask, contours


def get_pxray_patella(just_patella):
    #print("here")
    
    #just_patella = cv2.threshold(MRI_to_Xray(patella_volume), 0.01,255,0)[1]
    
    #just_patella = cv2.cvtColor(just_patella, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(just_patella,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    contours =[max(contours, key = cv2.contourArea)]   #deals with case where multiplec contours detected but ASSUMES BIGGEST IS PATELLA
    # Draw all contours 
    # -1 signifies drawing all contours 
    contour_mask = np.zeros(just_patella.shape)
    cv2.drawContours(contour_mask, contours, -1, 255, 3) 
    #plt.imshow(contour_mask)
    #plt.show()
    #print(contour_mask.shape)
    #contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_RGB2GRAY)
    return contour_mask, contours





def get_inital_scale_factor(xray, xray_patella_contours, pxray, pxray_patella_contours):
    print("Number of xray contours passed is ", len(xray_patella_contours))
    print("Number of pxray contours passed is ", len(pxray_patella_contours))
    assert (len(xray_patella_contours)==1 and len(pxray_patella_contours)==1)
    
    xray_rect = cv2.boundingRect(xray_patella_contours[0])
    pxray_rect = cv2.boundingRect(pxray_patella_contours[0])
    width_scale_factor = pxray_rect[2] / xray_rect[2]
    height_scale_factor = pxray_rect[3] / xray_rect[3]

    width_scale_factor = max(width_scale_factor, height_scale_factor) #keeping aspect ratio the same
    height_scale_factor = width_scale_factor
    scale_factor = height_scale_factor
    
    return scale_factor, xray_rect, pxray_rect

def scale(xray, scale_factor, xray_rect, pxray_rect):
    scaled_xray = cv2.resize(xray, (0,0), fx = scale_factor, fy = scale_factor)
    pxray_centre = (int(pxray_rect[0]+ (pxray_rect[2]//2)), int(pxray_rect[1]+ (pxray_rect[3]//2)))
    xray_centre = (int(xray_rect[0]*scale_factor+ (xray_rect[2]*scale_factor//2)), int(xray_rect[1]*scale_factor+ (xray_rect[3]*scale_factor//2)))
    return scaled_xray, pxray_centre, xray_centre

def scale_initial(xray, xray_patella_contours, pxray, pxray_patella_contours):
    scale_factor, xray_rect, pxray_rect = get_inital_scale_factor(xray, xray_patella_contours, pxray, pxray_patella_contours)
    return scale(xray, scale_factor, xray_rect, pxray_rect)




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

    cropped_xray = scaled_xray


    left_bound = xray_centre[0]-pxray_centre[0]
    right_bound = left_bound + pxray.shape[1]
    top_bound = xray_centre[1]-pxray_centre[1]
    bottom_bound = top_bound+pxray.shape[0]
    #print(top_bound, bottom_bound, left_bound, right_bound)

    if left_bound < 0:
        print("leftbound was too small")
        padding_needed = abs(left_bound)
        cropped_xray = np.transpose(cropped_xray)
        cropped_xray = np.concatenate([[[0 for i in range(0,cropped_xray.shape[1])] for j in range (0,padding_needed)], cropped_xray])
        cropped_xray = np.transpose(cropped_xray)
        left_bound = 0
        right_bound += padding_needed
    if right_bound >= scaled_xray.shape[1]:
        print("rightbound was too big")
        print("old xray shape: ", cropped_xray.shape)
        print("old right bound: ", right_bound)
        padding_needed  = right_bound - scaled_xray.shape[1] +1
        cropped_xray = np.transpose(cropped_xray)
        cropped_xray = np.concatenate([cropped_xray, np.array([[0 for i in range(0,cropped_xray.shape[1])] for j in range (0,padding_needed)])])
        cropped_xray = np.transpose(cropped_xray)
        
        print("New xray dimensions ", cropped_xray.shape)
        print("New right bound: ", right_bound)
    if top_bound < 0:
        padding_needed =  abs(top_bound)
        cropped_xray = np.concatenate([[[0 for i in range(0,cropped_xray.shape[1])] for j in range (0,padding_needed)], cropped_xray])
        top_bound = 0
        bottom_bound += padding_needed
        print("top bound was too small")
    if bottom_bound >= scaled_xray.shape[0]:
        padding_needed = bottom_bound -scaled_xray.shape[0] +1
        cropped_xray = np.concatenate([cropped_xray, [[0 for i in range(0,cropped_xray.shape[1])] for j in range (0,padding_needed)]])
        print("bottom bound was too big")


    pre_cropped_xray = cropped_xray #this is silly naming
    cropped_xray = cropped_xray[top_bound:bottom_bound, left_bound:right_bound]
    overlay = 0.5*cropped_xray + 0.5*pxray
    #plt.imshow(overlay, cmap = 'gray')
    #plt.show()
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
    #fg,ax = plt.subplots(1,2)
    #ax[0].imshow(cropped_xray, cmap = 'gray', vmin = 0, vmax = 255)
    #ax[1].imshow(pxray, cmap= 'gray', vmin = 0, vmax = 255)
    #plt.show()

    

    

    return cropped_xray, top_bound, bottom_bound, left_bound, right_bound, pre_cropped_xray #pre_cropped_xray is the newly scaled xray to take account for bound problems

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
        just_patella = np.where(MRI_to_Xray(rotated_patella_volume)>= 0.01,255,0).astype(np.uint8)
        contour_mask, pxray_contours = get_pxray_patella(just_patella)


        
        rotated_volume = rotate_volume(volume, angle, without_cartilidge=True)
        pxray = 255*MRI_to_Xray(rotated_volume)

        
        scaled_xray, pxray_centre, xray_centre = scale_initial(xray, xray_contours[2:3], pxray, pxray_contours)
        cropped_xray ,top_bound, bottom_bound, left_bound, right_bound, pre_cropped_xray= crop_scaled_xray(scaled_xray, pxray_centre, xray_centre, pxray)
        print(top_bound, left_bound)

        overlay = 0.5*cropped_xray + 0.5*pxray
        #plt.imshow(overlay, cmap ='gray')
        #plt.show()
        

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

        #overlay = 0.5*cropped_xray + 0.5*pxray
        #plt.imshow(overlay, cmap ='gray')
        #plt.show()
        #difference = cropped_xray - pxray
        """
        fg, ax = plt.subplots(2,2)
        fg.suptitle("Height " + str(top))
        ax[0][0].imshow(cropped_xray, cmap= 'gray')
        ax[0][0].set_title("Pseudo_xray")
        ax[0][1].imshow(pxray, cmap= 'gray')
        ax[0][1].set_title("Cropped_xray")
        ax[1][0].imshow(overlay, cmap = 'gray')
        ax[1][0].set_title("Overlay")
        ax[1][1].imshow(difference, cmap = 'gray')
        ax[1][1].set_title("Difference image")
        plt.show()
        """
        


        entropy = entropy_of_difference(cropped_xray, pxray, scalar, num_bins)

        entropies.append(entropy)
        print("Entropy of  this angle at height ", top, "is ", entropy)


        top += step
        i+=1
    print("height entopies are ", entropies)
    print("heights are ", heights)
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
        just_patella = np.where(MRI_to_Xray(rotated_patella_volume)>= 0.01,255,0).astype(np.uint8)
        contour_mask, pxray_contours = get_pxray_patella(just_patella)


        
        rotated_volume = rotate_volume(volume, angle, without_cartilidge=True)
        pxray = 255*1.3*MRI_to_Xray(rotated_volume)

        
        scaled_xray, pxray_centre, xray_centre = scale_initial(xray, xray_contours, pxray, pxray_contours)
        cropped_xray ,top_bound, bottom_bound, left_bound, right_bound, scaled_xray = crop_scaled_xray(scaled_xray, pxray_centre, xray_centre, pxray)

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



def rot_scale_vert(volume, rot_start, rot_num_images, rot_step, xray, xray_patella_mask, scalar, num_bins, num_up, num_down, trans_step, num_inflate, num_deflate, resize = 0.5):
    patella_volume = cv2.threshold((cv2.threshold(volume,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
    xray_patella_mask, xray_contours = get_xray_patella(xray_patella_mask)
    #contoured_xray = cv2.drawContours(xray, xray_contours, -1, 255, 1)
    #plt.imshow(contoured_xray)
    #plt.show()
    #xray_contour = xray_contours[2:3] #for 9911221 , need to fix
    print(len(xray_contours))
    #xray_contour = xray_contours[1:2] # for 9947240 need to fix
    angle_entropies = []
    angles = []

    for i in range (0, rot_num_images):
        
        angle = (rot_start + rot_step*i)%360
        angles.append(angle)
        print("Angle ", angle)
        print("------------------------")

    
        rotated_patella_volume = rotate_volume(patella_volume,angle)
        just_patella = cv2.resize(np.where(MRI_to_Xray(rotated_patella_volume)>= 0.01,255,0).astype(np.uint8), (0,0), fx=resize, fy=resize)
        contour_mask, pxray_contours = get_pxray_patella(just_patella)

        rotated_volume = rotate_volume(volume, angle, without_cartilidge=True)
        pxray = cv2.resize(255*1.3*MRI_to_Xray(rotated_volume) , (0,0), fx=resize, fy=resize) #THIS 1.3 is random and needs adjusting

        initial_scale, xray_rect, pxray_rect = get_inital_scale_factor(xray, xray_contours, pxray, pxray_contours)
        scale_factor = initial_scale

        scale_entropies = []
        scales = []

        for k in range (0, num_inflate):
            print("Scale factor ", str(scale_factor))
            print("-------------------------------")
            scaled_xray, pxray_centre, xray_centre = scale(xray, scale_factor, xray_rect, pxray_rect)
            cropped_xray ,top_bound, bottom_bound, left_bound, right_bound , scaled_xray= crop_scaled_xray(scaled_xray, pxray_centre, xray_centre, pxray)
            overlay = cropped_xray*0.5 + pxray*0.5
            #plt.imshow(overlay, cmap = 'gray')
            #plt.show()
            print(top_bound, left_bound)
            print("Starting vertical translation entropies for angle ", str(angle), " and scale ", str(scale_factor))
            vert_entropies, heights = vertical_translational_freedom(scaled_xray, top_bound, left_bound, num_up, num_down, trans_step, pxray, scalar, num_bins)
            min_vert_entropy = np.min(vert_entropies)
            height_min_entropy = heights[np.argmin(vert_entropies)]
            scale_entropies.append((min_vert_entropy, height_min_entropy))
            print("Entopy for angle ", str(angle), " and scale ", str(scale_factor), " is ", str(min_vert_entropy))
            scales.append(scale_factor)
            scale_factor += 0.1

        scale_factor= initial_scale -0.1
        for m in range (0, num_deflate):
            print("Scale factor ", str(scale_factor))
            print("-------------------------------")
            scaled_xray, pxray_centre, xray_centre = scale(xray, scale_factor, xray_rect, pxray_rect)
            cropped_xray ,top_bound, bottom_bound, left_bound, right_bound, scaled_xray = crop_scaled_xray(scaled_xray, pxray_centre, xray_centre, pxray)

            print(top_bound, left_bound)
            print("Starting vertical translation entropies for angle ", str(angle), " and scale ", str(scale_factor))
            vert_entropies, heights = vertical_translational_freedom(scaled_xray, top_bound, left_bound, num_up, num_down, trans_step, pxray, scalar, num_bins)
            min_vert_entropy = np.min(vert_entropies)
            height_min_entropy = heights[np.argmin(vert_entropies)]
            scale_entropies.append((min_vert_entropy, height_min_entropy))
            print("Entopy for angle ", str(angle), " and scale ", str(scale_factor), " is ", str(min_vert_entropy))
            scales.append(scale_factor)
            scale_factor -= 0.1


        

        #overlay = 0.5*cropped_xray + 0.5*pxray
        #plt.imshow(overlay, cmap ='gray')
        #plt.show()
        
        print("scale entropies are ", scale_entropies)
        print("scales are: ", scales)
        angle_entropy, height_of_min_entropy = min (scale_entropies)
        scale_of_min_entropy = scales[scale_entropies.index((angle_entropy, height_of_min_entropy))]
        angle_entropies.append((angle_entropy, scale_of_min_entropy, height_of_min_entropy))
        print("Minimum entropy for angle ", str(angle), " is ", angle_entropy, " at scale ", scale_of_min_entropy, " and height ", height_of_min_entropy)
    
    print("angle entropies", angle_entropies)
    print("angles: ", angles)
    min_angle_entropy, scale_achieved_at, height_achieved_at = min(angle_entropies)
    angle_achieved_at = angles[angle_entropies.index((min_angle_entropy, scale_achieved_at, height_achieved_at))]
    
    return angle_entropies, min_angle_entropy, angle_achieved_at, scale_achieved_at, height_achieved_at



def registration_experiment(patient, side, rot_start, rot_num_images, rot_step,scalar, num_bins, num_up, num_down, trans_step, num_inflate, num_deflate, xray_patella_centre, tibial_coordinates, kernel_size, resize =0.5):
    _, volume = loadPatientMask(patient, side, "1")
    xray_patella_mask = cv2.imread("xrays\\"+patient+"_"+side.lower()+"_xray_contour.png", cv2.IMREAD_GRAYSCALE)
    original_xray = cv2.imread("xrays\\"+patient+"_"+side.lower()+"_xray.jpg", cv2.IMREAD_GRAYSCALE)
    #xray_patella_centre = (xray_patella_centre[0]*resize, xray_patella_centre[1]*resize)
    #tibial_coordinates = (tibial_coordinates[0]*resize, tibial_coordinates[1]*resize)
    xray = pre_process_xray_tibialvals(original_xray, xray_patella_mask, xray_patella_centre, tibial_coordinates, kernel_size)
    xray = cv2.resize(xray, (0,0), fx=resize, fy =resize)
    xray_patella_mask =cv2.resize(xray_patella_mask, (0,0), fx=resize, fy =resize)
    np.save("results\\"+patient+"_"+side+"_enhanced_xray.npy", xray)
    entropies, min_entropy, angle_achieved_at, scale_achieved_at, height_achieved_at = rot_scale_vert(volume, rot_start, rot_num_images,rot_step ,xray, xray_patella_mask, scalar, num_bins, num_up, num_down, trans_step, num_inflate, num_deflate, resize)
    print(entropies)
    print("Min entropy was ", min_entropy, " achieved at (angle, scale, height) ", angle_achieved_at, scale_achieved_at, height_achieved_at)
    np.savez_compressed("results\\"+patient+"_"+side+"_angleEntropies.npz", e= entropies, s = [min_entropy, angle_achieved_at, scale_achieved_at, height_achieved_at])
    
    #xray = cv2.imread("xrays\\"+patient+"_"+side.lower()+"_xray.jpg", cv2.IMREAD_GRAYSCALE) #unprocessing the xray for sake of display
    patella_volume = cv2.threshold((cv2.threshold(volume,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
    xray_patella_mask, xray_contours = get_xray_patella(xray_patella_mask)
    rotated_patella_volume = rotate_volume(patella_volume,angle_achieved_at)
    just_patella = cv2.resize(np.where(MRI_to_Xray(rotated_patella_volume)>= 0.01,255,0).astype(np.uint8),(0,0), fx=resize, fy=resize)
    contour_mask, pxray_contours = get_pxray_patella(just_patella)

    rotated_volume = rotate_volume(volume, angle_achieved_at, without_cartilidge=True)
    
    pxray = cv2.resize(255*1.3*MRI_to_Xray(rotated_volume) , (0,0), fx=resize, fy=resize) #THIS 1.3 is random and needs adjusting
    initial_scale, xray_rect, pxray_rect = get_inital_scale_factor(xray, xray_contours, pxray, pxray_contours)
    scale_factor = scale_achieved_at
    scaled_xray, pxray_centre, xray_centre = scale(xray, scale_factor, xray_rect, pxray_rect)
    cropped_xray ,top_bound, bottom_bound, left_bound, right_bound , scaled_xray= crop_scaled_xray(scaled_xray, pxray_centre, xray_centre, pxray)
    top_bound = height_achieved_at
    cropped_xray = scaled_xray[top_bound:top_bound+pxray.shape[0], left_bound: right_bound]
    overlay = 0.6*cropped_xray + 0.4*pxray
    difference = cropped_xray-pxray
    #plt.imshow(overlay, cmap='gray')
    #plt.show()
    
    fg, ax = plt.subplots(2,2)
    fg.suptitle("Angle " + str(angle_achieved_at) + ", Scale " + str(scale_achieved_at)+ ", Height " + str(height_achieved_at))
    ax[0][0].imshow(cropped_xray, cmap= 'gray')
    ax[0][0].set_title("Enhanced_xray")
    ax[0][1].imshow(pxray, cmap= 'gray')
    ax[0][1].set_title("Pseudo_xray")
    ax[1][0].imshow(overlay, cmap = 'gray')
    ax[1][0].set_title("Overlay")
    ax[1][1].imshow(difference, cmap = 'gray')
    ax[1][1].set_title("Difference image")
    #plt.show()

    plt.savefig("results\\"+patient+"_"+side+"_registered_images.png")








patients = ["9002316", "9002411", "9002817", "9911221", "9911721", "9917307", "9918802", "9921811", "9924274", "9947240"]
patient_tib_centres = [(170,300), (170,300),(170, 270), (170,303),(170,239), (185,325), (121,286), (179,321),(207,312) , (170,339)]
patient_pat_centres = [(170,220), (170,196), (170,186), (155,232),(182,169), (170,200), (103,195), (178,226), (199, 208), (169,238)]

for i in range (5, len(patients)):
    print(i)
    registration_experiment(patients[i], "LEFT", 348,6,4, 1, 64, 6, 6, 4, 4, 4, patient_pat_centres[i], patient_tib_centres[i], 32)









