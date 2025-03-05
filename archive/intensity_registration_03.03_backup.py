import numpy as np
from newRotateForMasks import rotate_volume
from MRI_to_Xray import MRI_to_Xray
from matplotlib import pyplot as plt
import cv2
from readMLMasks import loadPatientMask
import random

"""
#old version of the function we keep for reference until final software complete

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

#**************************************IMAGE PREPARATION FUNCTIONS**********************************************

#takes in an xray and some clinican input and outputs an enahnced xray ready to be registerred
def pre_process_xray_tibialvals(xray_image, xray_patella_mask, xray_patella_centre, tibial_coordinates, kernel_size, save = False, save_location = ""):
    if save:
        filename = save_location+"original_xray.png"
        cv2.imwrite(filename, xray_image)
    PXRAY_TIBIAL_VAL  = 80 # THIS IS A CONSTANT DESIGNED TO MATCH THE TIBIAL VALUE ON PSEUDO-XRAYS

    blurred = np.copy(xray_image)
    blurred = cv2.blur(blurred, (11,11))  #more blur for use in adaptive thresholding
    xray_image = cv2.blur(xray_image, (4,4))


    thresholded_mask = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)
    if save:
        filename = save_location + "thresholded_mask.png"
        cv2.imwrite(filename, thresholded_mask)

    sample = xray_image[tibial_coordinates[1]-(kernel_size//2): tibial_coordinates[1]+(kernel_size//2), tibial_coordinates[0]-(kernel_size//2):tibial_coordinates[0]+(kernel_size//2)]
    min_val = np.min(sample)
    
    xray_image = xray_image + (12/255)*thresholded_mask

    if save:
        filename = save_location + "pre_thresholding.png"
        cv2.imwrite(filename, xray_image)
    
    xray_image = cv2.threshold(xray_image, min_val, 255, cv2.THRESH_TOZERO)[1]
    xray_image = (PXRAY_TIBIAL_VAL / min_val) *xray_image

    if save:
        filename = save_location + "post_thresholding.png"
        cv2.imwrite(filename, xray_image)
    
    #now want to find the average patella brightness so that it can be appropriately enhanced.
    brightest_mask = np.where(xray_patella_mask >0, 0, 1)
    patella_sample = xray_image[xray_patella_centre[1]-(kernel_size//2): xray_patella_centre[1]+(kernel_size//2), xray_patella_centre[0]-(kernel_size//2):xray_patella_centre[0]+(kernel_size//2)]
    avg_patella_val = np.average(patella_sample)


    PXRAY_PAT_AVG = 110 #THIS IS A CONSTANT DESIGNED TO MATCH THE PATELLA VALUE ON PSEUDO-XRAYS
    amount_brightness_to_add = PXRAY_PAT_AVG- avg_patella_val
    xray_image = xray_image + amount_brightness_to_add*brightest_mask

    if save:
        filename = save_location + "post_enhancement.png"
        cv2.imwrite(filename, xray_image)
    return xray_image


#prepares a pseudoxray with the appropriate colours for tibia and patella. Note the chosen values should match the constants defined in pre_proces_xray_tibial_values
def pre_process_pxray(rotated_volume, patella_mask, chosen_tibial_value, chosen_patella_value, resize):
    patella_mask = np.where(patella_mask>0, 1, 0) #just in case patella_mask is given as 255 or 0 rather than as 1 or 0 
    base_pxray = cv2.resize(MRI_to_Xray(rotated_volume), (0,0), fx=resize, fy=resize)
    binary_pxray = chosen_tibial_value*np.where(base_pxray >0,1, 0)
    amount_to_add = chosen_patella_value - chosen_tibial_value
    final_xray = binary_pxray + amount_to_add*patella_mask
    return final_xray



def get_xray_patella(xray_patella_mask, contour_colour = "black"):
    if contour_colour=="black":
        xray_patella_mask = (255 - xray_patella_mask)
    contours, hierarchy = cv2.findContours(xray_patella_mask,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    contours =[max(contours, key = cv2.contourArea)]  #validation; ensures that len(contours) =1 and this contour is that of the patella
    return xray_patella_mask, contours


#takes in the patella mask and finds it's contour
def get_pxray_patella(just_patella):

    contours, hierarchy = cv2.findContours(just_patella,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    contours =[max(contours, key = cv2.contourArea)]   #deals with case where multiplec contours detected but ASSUMES BIGGEST IS PATELLA
    # Draw all contours 
    # -1 signifies drawing all contours 
    contour_mask = np.zeros(just_patella.shape)
    cv2.drawContours(contour_mask, contours, -1, 255, 3) 
    #plt.imshow(contour_mask)
    #plt.show()

    return contour_mask, contours

#uses relative size of pxray patella and xray patella to get an approximate scale factor for registering the size of the two scans
def get_inital_scale_factor(xray, xray_patella_contours, pxray, pxray_patella_contours):
    assert (len(xray_patella_contours)==1 and len(pxray_patella_contours)==1)
    
    xray_rect = cv2.boundingRect(xray_patella_contours[0])
    pxray_rect = cv2.boundingRect(pxray_patella_contours[0])
    width_scale_factor = pxray_rect[2] / xray_rect[2]
    height_scale_factor = pxray_rect[3] / xray_rect[3]

    width_scale_factor = max(width_scale_factor, height_scale_factor) #keeping aspect ratio the same
    height_scale_factor = width_scale_factor
    scale_factor = height_scale_factor
    
    return scale_factor, xray_rect, pxray_rect

#scales the xray by scale_factor and determines the (new) centres of the scaled xray and pxray s.t. they can later be aligned
def scale(xray, scale_factor, xray_rect, pxray_rect):
    scaled_xray = cv2.resize(xray, (0,0), fx = scale_factor, fy = scale_factor)
    pxray_centre = (int(pxray_rect[0]+ (pxray_rect[2]//2)), int(pxray_rect[1]+ (pxray_rect[3]//2)))
    xray_centre = (int(xray_rect[0]*scale_factor+ (xray_rect[2]*scale_factor//2)), int(xray_rect[1]*scale_factor+ (xray_rect[3]*scale_factor//2)))
    return scaled_xray, pxray_centre, xray_centre

#does the inital rough registration of xray to pxray
def scale_initial(xray, xray_patella_contours, pxray, pxray_patella_contours):
    scale_factor, xray_rect, pxray_rect = get_inital_scale_factor(xray, xray_patella_contours, pxray, pxray_patella_contours)
    return scale(xray, scale_factor, xray_rect, pxray_rect)

#returns the section of the xray which overlaps with pxray when they are aligned by their centres
def crop_scaled_xray(scaled_xray, pxray_centre, xray_centre, pxray):

    cropped_xray = scaled_xray

    left_bound = xray_centre[0]-pxray_centre[0]
    right_bound = left_bound + pxray.shape[1]
    top_bound = xray_centre[1]-pxray_centre[1]
    bottom_bound = top_bound+pxray.shape[0]
    
    #below if statements pads the image where necessary in order to overlay them on the same image
    if left_bound < 0:
        
        padding_needed = abs(left_bound)
        cropped_xray = np.transpose(cropped_xray)
        cropped_xray = np.concatenate([[[0 for i in range(0,cropped_xray.shape[1])] for j in range (0,padding_needed)], cropped_xray])
        cropped_xray = np.transpose(cropped_xray)
        left_bound = 0
        right_bound += padding_needed
    if right_bound >= scaled_xray.shape[1]:

        padding_needed  = right_bound - scaled_xray.shape[1] +1
        cropped_xray = np.transpose(cropped_xray)
        cropped_xray = np.concatenate([cropped_xray, np.array([[0 for i in range(0,cropped_xray.shape[1])] for j in range (0,padding_needed)])])
        cropped_xray = np.transpose(cropped_xray)

    if top_bound < 0:
        padding_needed =  abs(top_bound)
        cropped_xray = np.concatenate([[[0 for i in range(0,cropped_xray.shape[1])] for j in range (0,padding_needed)], cropped_xray])
        top_bound = 0
        bottom_bound += padding_needed

    if bottom_bound >= scaled_xray.shape[0]:
        padding_needed = bottom_bound -scaled_xray.shape[0] +1
        cropped_xray = np.concatenate([cropped_xray, [[0 for i in range(0,cropped_xray.shape[1])] for j in range (0,padding_needed)]])



    pre_cropped_xray = cropped_xray #this is silly naming
    cropped_xray = cropped_xray[top_bound:bottom_bound, left_bound:right_bound]
    #overlay = 0.5*cropped_xray + 0.5*pxray
    #plt.imshow(overlay, cmap = 'gray')
    #plt.show()
    #fg,ax = plt.subplots(1,2)
    #ax[0].imshow(cropped_xray, cmap = 'gray', vmin = 0, vmax = 255)
    #ax[1].imshow(pxray, cmap= 'gray', vmin = 0, vmax = 255)
    #plt.show()

    return cropped_xray, top_bound, bottom_bound, left_bound, right_bound, pre_cropped_xray #pre_cropped_xray is the newly scaled xray to take account for bound problems

#******************************SIMILARITY METRICS****************************************************************


#https://kclpure.kcl.ac.uk/ws/portalfiles/portal/12065027/Studentthesis-Graeme_Penney_2000.pdf
#make sure inputs are numpy arrays and not just lists
def entropy_of_difference(image1, image2, scale,num_bins):
    """
    #old implementation below but good for demonstrating what is actually going on here
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
    """

    difference_array = image1 - scale*image2

    hist, edges = np.histogram(difference_array, bins=num_bins)
    px = hist/ float(np.sum(hist))
    nx = px >0
    entropy  = -1*np.sum(px[nx]*np.log(px[nx]))
    return entropy



#this makes any black areas in the difference image noisy such that the entropy is higher
def entropy_with_penalities(image1, image2, scale,num_bins, noise_constant = 50):
    assert(image1.shape == image2.shape)

    difference_array = image1 - scale*image2

    (minVal, maxVal) = (np.min(difference_array), np.max(difference_array))

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
            if (abs(value-minVal)) < 5: 
                value = value + np.random.randint(0, noise_constant)
                difference_array[i][j] =value
            bin = find_index_of_bin(value)
            bin_counts[bin] +=1
    
    total_num_values = difference_array.shape[0]*difference_array.shape[1]
    entropy = 0
    for bin in range(0, num_bins):
        prob = bin_counts[bin]/total_num_values
        if prob != 0:
            entropy += prob*np.log(prob)
    entropy = -1*entropy
    #plt.imshow(difference_array, cmap = 'gray')
    #plt.show()

    return entropy

def mutual_information(image1, image2, num_bins):

    hist_2d, x_edges, y_edges = np.histogram2d(
    image1.ravel(),
    image2.ravel(),
    bins=num_bins)

    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum

    return -1*np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def gxy(image1, image2, num_bins, kernel=3):

    image1 = np.array(image1, np.uint16)
    image2 = np.array(image2, np.uint16)
    image1_gX = cv2.Sobel(image1, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=kernel)
    image1_gY = cv2.Sobel(image1, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=kernel)
    image2_gX = cv2.Sobel(image2, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=kernel)
    image2_gY = cv2.Sobel(image2, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=kernel)

    gxy = 0

    for x in range(0,image1.shape[0]):
        for y in range(0,image1.shape[0]):
            
    
            vec1 = [image1_gX[x][y], image1_gY[x][y]]
        

            vec2 = [image2_gX[x][y], image2_gY[x][y]]


            minvals = min(np.linalg.norm(vec1),np.linalg.norm(vec2))

            if minvals != 0:
   
                internal =  np.round(np.dot(vec1,vec2) /(np.linalg.norm(vec1)*np.linalg.norm(vec2)),6)
             
                alpha = np.arccos(internal )

                wa = (np.cos(2*alpha)+1)/2
      
                gxy+= wa*minvals


    return gxy

#pre: image 1 is xray, image2 is pxray
#this is a more efficient version which only uses gradient information for pixels on the contour of the pxray
def mygxy(image1, image2, num_bins, kernel = 3):
    image1 = np.array(image1, np.uint8)
    image2 = np.array(image2, np.uint8)
    image1_gX = cv2.Sobel(image1, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=kernel)
    image1_gY = cv2.Sobel(image1, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=kernel)
    image2_gX = cv2.Sobel(image2, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=kernel)
    image2_gY = cv2.Sobel(image2, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=kernel)


    contours, hierarchy = cv2.findContours(image2,  #ASSUMES image2 is pxray 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    pixels_to_search = np.concatenate(np.concatenate(list(contours)))

    gxy = 0


    for (y,x) in pixels_to_search:
        vec1 = [image1_gX[x][y], image1_gY[x][y]]

        vec2 = [image2_gX[x][y], image2_gY[x][y]]

    
        minvals = min(np.linalg.norm(vec1),np.linalg.norm(vec2))

        if minvals != 0:

            internal =  np.round(np.dot(vec1,vec2) /(np.linalg.norm(vec1)*np.linalg.norm(vec2)),6)
            alpha = np.arccos(internal )

            wa = (np.cos(2*alpha)+1)/2


            gxy+= wa*minvals

    return gxy





def GMI(image1, image2, num_bins, kernel =3):

    mutualinf = mutual_information(image1, image2, num_bins)
    gxyval = gxy(image1, image2, num_bins, kernel)
    return gxyval*mutualinf

def GMI_e(image1, image2, num_bins, kernel =3):

    mutualinf = mutual_information(image1, image2, num_bins)
    gxyval = mygxy(image1, image2, num_bins, kernel)
    return gxyval*mutualinf

def gradient_difference(xray, pxray, scalar):
    xray = np.array(xray, np.uint8)
    pxray = np.array(pxray, np.uint8)
    xray_gX = cv2.Sobel(xray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    xray_gY = cv2.Sobel(xray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    pxray_gX = cv2.Sobel(pxray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    pxray_gY = cv2.Sobel(pxray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    diff_vertical = xray_gX - scalar*pxray_gX
    diff_horizontal = xray_gY - scalar*pxray_gY
    Av = np.var(diff_vertical)
    Ah = np.var(diff_horizontal)

    gradient_difference_vert = 0
    gradient_difference_hor = 0
    for i in range (0, pxray.shape[0]):
        for j in range (0, pxray.shape[1]):
            gradient_difference_vert += Av/ (Av + diff_vertical[i][j]**2)
            gradient_difference_hor += Ah/ (Ah + diff_horizontal[i][j]**2)
    return -1*(gradient_difference_vert + gradient_difference_hor)



#**********************************************RESULTS UTILITIES**********************************************************


def plot_entropies(volume, start, num_images, step, xray, scale, num_bins):
    entropies = []

    for i in range (0, num_images):
        angle = start + step*i
        
        rotated_volume = rotate_volume(volume, angle, alreadyPadded=True)
        pseudo_xray = MRI_to_Xray(rotated_volume)
        
        entropy = entropy_of_difference(xray, pseudo_xray, scale, num_bins)
        entropies.append(entropy)
    
    return entropies


#***********************************************************EXPERIMENTS / REGISTRATION PROCEDURES ***********************************


#this is when we only allow rotational degree of freedom but I think it's out of date
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
def vertical_translational_freedom(xray, top_bound, left_bound, num_up, num_down, step, pxray, scalar, num_bins,  angle, scale_factor, measure = 'entropy', noise_constant =10,):
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
        """
        overlay = 0.5*cropped_xray + 0.5*pxray
        #plt.imshow(overlay, cmap ='gray', vmin = 0, vmax= 255)
        #plt.show()
        difference = cropped_xray - pxray
        
        fg, ax = plt.subplots(1,2)
        #rotation_relative_to_true_coronal = (angle+angle_achieved_at)%360
        #if rotation_relative_to_true_coronal >180:
        #    rotation_relative_to_true_coronal = -1*(360-rotation_relative_to_true_coronal)
        
        fg.suptitle("Pure rot angle " + str(angle) + ", Scale " + str(scale_factor)+ ", Height " + str(top))
        ax[0].imshow(overlay, cmap = 'gray')
        ax[0].set_title("Overlay")
        ax[1].imshow(difference, cmap = 'gray')
        ax[1].set_title("Difference image")
        #plt.show()

        plt.savefig("results\\temp\\"+str(angle)+"_"+str(round(scale_factor,2))+"_"+str(round(top,2))+".png")

        plt.close(fg)
        
        """



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
        

        if measure == 'entropy':
            entropy = entropy_of_difference(cropped_xray, pxray, scalar, num_bins)
        elif measure == 'gradient_difference':
            entropy = gradient_difference(cropped_xray, pxray, scalar)
        elif measure == 'entropy_with_penalties':
            entropy = entropy_with_penalities(cropped_xray,pxray, scalar, num_bins, noise_constant)
        elif measure == 'mutual_information':
            entropy = mutual_information(cropped_xray, pxray, num_bins)
        elif measure == 'gmi':
            entropy = GMI(cropped_xray, pxray, num_bins)

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



def rot_scale_vert(volume, rot_start, rot_num_images, rot_step, xray, xray_patella_mask, scalar, num_bins, num_up, num_down, trans_step, num_inflate, num_deflate, resize = 0.5, measure ='entropy', noise_constant =10):
    patella_volume = cv2.threshold((cv2.threshold(volume,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
    xray_patella_mask, xray_contours = get_xray_patella(xray_patella_mask)
    #contoured_xray = cv2.drawContours(xray, xray_contours, -1, 255, 1)
    #plt.imshow(contoured_xray)
    #plt.show()

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
        
        pxray = pre_process_pxray(rotated_volume, just_patella, 85, 115, resize)
        #plt.imshow(pxray, cmap = 'gray', vmin = 0, vmax =255)
        #plt.show()

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
            vert_entropies, heights = vertical_translational_freedom(scaled_xray, top_bound, left_bound, num_up, num_down, trans_step, pxray, scalar, num_bins,angle=angle, scale_factor=scale_factor, measure=measure, noise_constant=noise_constant )
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
            vert_entropies, heights = vertical_translational_freedom(scaled_xray, top_bound, left_bound, num_up, num_down, trans_step, pxray, scalar, num_bins, angle=angle, scale_factor=scale_factor, measure=measure, noise_constant=noise_constant )
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



def registration_experiment(patient, side, rot_start, rot_num_images, rot_step,scalar, num_bins, num_up, num_down, trans_step, num_inflate, num_deflate, xray_patella_centre, tibial_coordinates, kernel_size, angle, resize =0.5, measure = 'entropy', noise_constant =10):
    _, volume = loadPatientMask(patient, side, "1")
    xray_patella_mask = cv2.imread("xrays\\"+patient+"_"+side.lower()+"_xray_contour.png", cv2.IMREAD_GRAYSCALE)
    original_xray = cv2.imread("xrays\\"+patient+"_"+side.lower()+"_xray.jpg", cv2.IMREAD_GRAYSCALE)
    resized_first_xray = cv2.resize(original_xray, (0,0), fx=resize, fy =resize)  #to be saved to show base case 


 
    xray = pre_process_xray_tibialvals(original_xray, xray_patella_mask, xray_patella_centre, tibial_coordinates, kernel_size)
    xray = cv2.resize(xray, (0,0), fx=resize, fy =resize)
    xray_patella_mask =cv2.resize(xray_patella_mask, (0,0), fx=resize, fy =resize)
    np.save("results\\"+patient+"_"+side+"_enhanced_xray.npy", xray)
    entropies, min_entropy, angle_achieved_at, scale_achieved_at, height_achieved_at = rot_scale_vert(volume, rot_start, rot_num_images,rot_step ,xray, xray_patella_mask, scalar, num_bins, num_up, num_down, trans_step, num_inflate, num_deflate, resize, measure=measure, noise_constant=noise_constant)
    print(entropies)
    print("Min entropy was ", min_entropy, " achieved at (angle, scale, height) ", angle_achieved_at, scale_achieved_at, height_achieved_at)
    np.savez_compressed("results\\"+patient+"_"+side+"_angleEntropies.npz", e= entropies, s = [min_entropy, angle_achieved_at, scale_achieved_at, height_achieved_at])
    f = open("temp_angles.text", "a")
    f.write(str(angle_achieved_at)+",")
    f.close()
   
    patella_volume = cv2.threshold((cv2.threshold(volume,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
    xray_patella_mask, xray_contours = get_xray_patella(xray_patella_mask)
    unrotated_just_patella =  cv2.resize(np.where(MRI_to_Xray(rotate_volume(patella_volume,0))>= 0.01,255,0).astype(np.uint8),(0,0), fx=resize, fy=resize)
    unrotated_pxray = pre_process_pxray(rotate_volume(volume,0, without_cartilidge=True), unrotated_just_patella, 85, 115, resize)


    rotated_patella_volume = rotate_volume(patella_volume,angle_achieved_at)
    just_patella = cv2.resize(np.where(MRI_to_Xray(rotated_patella_volume)>= 0.01,255,0).astype(np.uint8),(0,0), fx=resize, fy=resize)
    contour_mask, pxray_contours = get_pxray_patella(just_patella)



    rotated_volume = rotate_volume(volume, angle_achieved_at, without_cartilidge=True)
    
    
    pxray = pre_process_pxray(rotated_volume, just_patella, 85, 100, resize)  #was 85,115
    initial_scale, xray_rect, pxray_rect = get_inital_scale_factor(xray, xray_contours, pxray, pxray_contours)
    scale_factor = scale_achieved_at
    scaled_xray, pxray_centre, xray_centre = scale(xray, scale_factor, xray_rect, pxray_rect)
    scaled_unprocessed_xray, _, _ = scale(resized_first_xray, scale_factor, xray_rect, pxray_rect)
    cropped_xray ,top_bound, bottom_bound, left_bound, right_bound , scaled_xray= crop_scaled_xray(scaled_xray, pxray_centre, xray_centre, pxray)
    cropped_unprocessed_xray, _, _,_,_, scaled_unprocessed_xray = crop_scaled_xray(scaled_unprocessed_xray, pxray_centre, xray_centre, pxray)
    top_bound = height_achieved_at
    cropped_xray = scaled_xray[top_bound:top_bound+pxray.shape[0], left_bound: right_bound]
    cropped_unprocessed_xray = scaled_unprocessed_xray[top_bound:top_bound+pxray.shape[0], left_bound: right_bound]
    true_xray_overlay = scaled_unprocessed_xray
    true_xray_overlay[top_bound:top_bound+pxray.shape[0], left_bound: right_bound] = 0.3*pxray+0.7*cropped_unprocessed_xray


    overlay = 0.7*cropped_xray + 0.3*pxray
    difference = cropped_xray-pxray
    #plt.imshow(overlay, cmap='gray')
    #plt.show()
    
    fg, ax = plt.subplots(2,2)
    rotation_relative_to_true_coronal = (angle+angle_achieved_at)%360
    if rotation_relative_to_true_coronal >180:
        rotation_relative_to_true_coronal = -1*(360-rotation_relative_to_true_coronal)
    
    fg.suptitle("Pure rot " + str(angle_achieved_at) + ", Relative angle " + str(rotation_relative_to_true_coronal) + ", Scale " + str(scale_achieved_at)+ ", Height " + str(height_achieved_at))
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

    fg2, ax2 = plt.subplots(1,3)
    ax2[0].imshow(resized_first_xray, cmap ='gray', vmin =0, vmax =255)
    ax2[0].set_title("Original Xray")
    ax2[1].imshow(unrotated_pxray, cmap ='gray', vmin =0, vmax =255)
    ax2[1].set_title("Unrotated Pseudo-Xray")
    ax2[2].imshow(true_xray_overlay, cmap ='gray', vmin = 0, vmax = 255)
    ax2[2].set_title("Registered Pseudo-Xray to Xray")
    plt.savefig("results\\"+patient+"_"+side+"_unprocessed_images.png")
    


angles = {  "9911221_LEFT_1":10,
            "9911221_RIGHT_1":345,
            "9911721_LEFT_1":7,
            "9911721_RIGHT_1":352,
            "9912946_LEFT_1":11,
            "9912946_RIGHT_1":350,
            "9917307_LEFT_1":7,
            "9917307_RIGHT_1":353,
            "9918802_LEFT_1":8,
            "9918802_RIGHT_1":352,
            "9921811_LEFT_1":2,
            "9921811_RIGHT_1":356,
            "9924274_LEFT_1":8,
            "9924274_RIGHT_1":357,
            "9937239_LEFT_1":10,
            "9937239_RIGHT_1":350,
            "9938236_LEFT_1":1,
            "9938236_RIGHT_1":358,
            "9943227_LEFT_1":3,
            "9943227_RIGHT_1":353,
            "9947240_LEFT_1":10,
            "9947240_RIGHT_1":347,
            "9958234_LEFT_1":4,
            "9958234_RIGHT_1":356,
            "9964731_LEFT_1":9,
            "9964731_RIGHT_1": 353 ,
            "9002116_LEFT_1":5,

            "9986355_LEFT_1": 13,
            "9986355_RIGHT_1": 345,
            "9986838_LEFT_1":4,
            "9986838_RIGHT_1":355,
            "9988027_LEFT_1":8,
            "9988027_RIGHT_1":350,
            "9988186_LEFT_1":7,
            "9988186_RIGHT_1":350,
            "9988305_LEFT_1":0,
            "9988305_RIGHT_1":355,
            "9988820_LEFT_1":6,
            "9988820_RIGHT_1":353,

            "9988421_RIGHT_1":353,
            "9988891_LEFT_1":4,
            "9988891_RIGHT_1":346,
            "9988921_LEFT_1":9,
            "9988921_RIGHT_1":348,
            "9989309_LEFT_1":9,
            "9989309_RIGHT_1":348,
            "9989352_LEFT_1":2,
            "9989352_RIGHT_1":354,
            "9989700_LEFT_1":8,
            "9989700_RIGHT_1":347,
            "9990072_LEFT_1":4,
            "9990072_RIGHT_1": 356,
            "9990192_LEFT_1":3,

            "9990355_LEFT_1":8,
            "9991018_LEFT_1":6,
            "9991313_LEFT_1":11,
            "9991580_LEFT_1":6,
            "9986207_LEFT_1":16,
            "9030296_LEFT_1":11,
            "9030418_LEFT_1":10,
            "9030718_LEFT_1":7,
            "9030925_LEFT_1":6,
            "9031141_LEFT_1":7,
            "9031426_LEFT_1":8,
            "9031930_LEFT_1":8,
            "9031961_LEFT_1":3,
            "9000622_LEFT_1":4,
            "9002316_LEFT_1":4,
            "9002411_LEFT_1":14,
            "9002430_LEFT_1":9,
            "9002817_LEFT_1":8,
            "9003126_LEFT_1":3,

            "9033937_LEFT_1":4,
            "9034451_LEFT_1": 7,
            "9034644_LEFT_1":0,
            "9034812_LEFT_1":7,
            "9034963_LEFT_1": 359,
            "9034991_LEFT_1":14,
            "9035317_LEFT_1":3,
            "9035647_LEFT_1":4,
            "9988421_LEFT_1":357,
            "9990698_LEFT_1":356,
            "9034677_LEFT_1":358,
            "9035449_LEFT_1":357
            
            }
                            


#PATIENT 9964731 is a very good example of an xray with a wildly misaligned patella



patients = ["9002316", "9002411", "9002817", "9911221", "9911721", "9917307", "9918802", "9921811", "9924274", "9947240", "9938236", "9943227", "9958234", "9964731", "9986355", "9986838", "9989352", "9989700", "9990192", "9990355", "9986207","9030925", "9031141", "9031930", "9031961", "9033937", "9034451", "9034677", "9034812", "9034963"]
patient_angles = []
for i in range (0, len(patients)):
    patient_angles.append(angles[patients[i]+"_LEFT_1"])
patient_tib_centres = [(170,300), (170,300),(170, 270), (170,303),(170,239), (185,325), (121,286), (179,321),(207,312) , (170,339), (181, 317), (203,326), (155,343),(183,269), (200,310), (158,307), (192,298), (170,300), (170,311), (179,338), (212,311),(192,335), (150,266), (120,339), (164,332), (178,350), (145,296), (156,256), (159,317), (209,329)]
patient_pat_centres = [(170,220), (170,196), (170,186), (155,232),(182,169), (170,200), (103,195), (178,226), (199, 208), (169,238), (174, 236), (206,240), (146,239), (209,172), (211,216), (153,198), (193,208), (178,206), (164,222), (170,201), (197,222), (204,158), (138,160), (100,166), (156,235), (170,197), (148,199), (151,167), (153,232), (209,238)]


gmi_step_4_results_corrected = [4, 10, 356, 10, 355,15,12,358,12,354,5,359,0,345,353,4,2,0,3,16,12,358,11,20,7,356,359,2,11,3] #these are indexed in roder of patients with none missing
gmi_step_4_results_raw = []
"""
with open("temp_angles.text", "r") as file:
    a = [line.strip() for line in file]

entropy_step_4_results_raw = list(map(int,a[0].split(",")))
#print(entropy_step_4_results_raw)
"""

for i in range(0,len(gmi_step_4_results_corrected)):#range(len(patients)):
    patient = patients[i]
    raw_angle = (gmi_step_4_results_corrected[i] - patient_angles[i]) %360
    gmi_step_4_results_raw.append(raw_angle)

mi_step_4_results_raw = [356,356,348,0,348,8,4,356,4,352,4,352, 356,336,340,0,356,352,0, 0,356,348,4,8,4,356,356,0,0,0]

#patient_index = [10,11,12,13,14,15,16,17,18,19,20,26,27,28,29]
#patient_index = [26,27,28,29]
#patient_index = [patients.index("9990355")]
patient_index = range(0, len(patients))
#patient_index = [10,11,12,13,14,15,16,17,18,19,20]


for i in patient_index:
    print(i)
    registration_experiment(patients[i], "LEFT", gmi_step_4_results_raw[i]-3,8, 1,1, 64, 12, 12, 2, 3, 3, patient_pat_centres[i], patient_tib_centres[i], 32, patient_angles[i], measure = 'gmi', noise_constant = 255)
    #registration_experiment(patients[i], "LEFT", 336,11, 4,1, 64, 3, 3, 4, 2, 2, patient_pat_centres[i], patient_tib_centres[i], 32, patient_angles[i], measure = 'entropy', noise_constant = 255)
    #registration_experiment(patients[i], "LEFT", mi_step_4_results_raw[i]-3,8, 1,1, 64, 12, 12, 2, 3, 3, patient_pat_centres[i], patient_tib_centres[i], 32, patient_angles[i], measure = 'mutual_information', noise_constant = 255)
    #registration_experiment(patients[i], "LEFT", entropy_step_4_results_raw[i]-3,8, 1,1, 64, 12, 12, 2, 3, 3, patient_pat_centres[i], patient_tib_centres[i], 32, patient_angles[i], measure = 'entropy', noise_constant = 255)
#was 336,11,4, 1, 64, 3, 3, 4, 2, 2,
#_,_, 1,1, 64, 12, 12, 2, 3, 3,  - for refined search

#gradient different isn't good here. Entropy needs some improving. Come back to this.



