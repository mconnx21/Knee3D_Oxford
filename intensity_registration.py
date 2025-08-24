import numpy as np
from newRotateForMasks import rotate_volume
from MRI_to_Xray import MRI_to_Xray
from matplotlib import pyplot as plt
import cv2
from readMLMasks import loadPatientMask
import random
import math
import time
import csv
import os


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

#this is GMIe
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



#given the original xray, the current position of it, and the pxray, test different vertical translations
def vertical_translational_freedom(xray, top_bound, left_bound, num_up, num_down, step, pxray, scalar, num_bins,  angle, scale_factor, measure = 'entropy', noise_constant =10, save_time = False):
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

        cropped_xray = xray[top:(top+h), left_bound:(left_bound+w)] 

        """
        #commented code below allows us to generate pics at every tested psoition - we use this for manually inspecting best registration
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


        start_coefficient = time.time()

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
        elif measure == 'gmi_e':
            entropy = GMI_e(cropped_xray, pxray, num_bins)

        end_coefficient =time.time()
        time_taken  = end_coefficient-start_coefficient
        
        if save_time==True:
            thisrecord = {}
            thisrecord["Number"] = i
            thisrecord["Time"] =time_taken

            with open("csvs//time_"+similarity_measure + ".csv", "a", newline = '') as csvfile:
                fieldnames = ["Number","Time"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(thisrecord)

        
        

        entropies.append(entropy)
        print("Entropy of  this angle at height ", top, "is ", entropy)


        top += step
        i+=1
    print("height entopies are ", entropies)
    print("heights are ", heights)
    return entropies, heights

#tests three degrees of freedom: rotation scale factor and vertical translation
def rot_scale_vert(volume, rot_start, rot_num_images, rot_step, xray, xray_patella_mask, scalar, num_bins, num_up, num_down, trans_step, num_inflate, num_deflate, resize = 0.5, measure ='entropy', noise_constant =10, gradient_descent = False):
    patella_volume = cv2.threshold((cv2.threshold(volume,2,7,cv2.THRESH_TOZERO))[1], 3,7, cv2.THRESH_TOZERO_INV)[1] #3 in the array
    xray_patella_mask, xray_contours = get_xray_patella(xray_patella_mask)
    #contoured_xray = cv2.drawContours(xray, xray_contours, -1, 255, 1)
    #plt.imshow(contoured_xray)
    #plt.show()

    angle_entropies = []
    angles = []
    done = False
    angle = rot_start
    images_done =0
    decay =1

    while not done:
        
        angles.append(angle)
        print("Angle ", angle)
        print("------------------------")

    
        rotated_patella_volume = rotate_volume(patella_volume,angle)
        just_patella = cv2.resize(np.where(MRI_to_Xray(rotated_patella_volume)>= 0.01,255,0).astype(np.uint8), (0,0), fx=resize, fy=resize)
        contour_mask, pxray_contours = get_pxray_patella(just_patella)

        rotated_volume = rotate_volume(volume, angle, without_cartilidge=True)
        
        pxray = pre_process_pxray(rotated_volume, just_patella, 85, 115, resize)
        #print(pxray.shape)
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
        #f = open("temp_angles.text", "a")
        #f.write(str(scales)+"\n")
        #f.write(str(scale_entropies)+"\n")
        #f.close()
        angle_entropy, height_of_min_entropy = min (scale_entropies)
        scale_of_min_entropy = scales[scale_entropies.index((angle_entropy, height_of_min_entropy))]
        angle_entropies.append((angle_entropy, scale_of_min_entropy, height_of_min_entropy))
        print("Minimum entropy for angle ", str(angle), " is ", angle_entropy, " at scale ", scale_of_min_entropy, " and height ", height_of_min_entropy)

        images_done+=1

        if not gradient_descent:

            angle = (angle +rot_step)%360
            
            if images_done >= rot_num_images:
                done = True
        
        else:
            
            if images_done<=1:
                angle = (angle -rot_step)%360
            
            else:

                last = angle_entropies[-1][0]
                secondlast= angle_entropies[-2][0]
                changeiny = last-secondlast
                lastx = angles[-1]
                if lastx<180:
                    lastx +=360
                secondlastx = angles[-2]
                if secondlastx<180:
                    secondlastx +=360
                changeinx = lastx-secondlastx

                gradient = changeiny /changeinx
                """
                step = -1* rot_step/gradient
                if abs(step) <1:
                    step = int(step/abs(step))

                """
                size = max(1, rot_step*decay)
                step = round(-1*np.sign(gradient)*size,0)
                
                

                angle = (angle +step)%360
            decay = decay*0.7
            if images_done >= rot_num_images:
                done = True




    print("angle entropies", angle_entropies)
    print("angles: ", angles)
    min_angle_entropy, scale_achieved_at, height_achieved_at = min(angle_entropies)
    angle_achieved_at = angles[angle_entropies.index((min_angle_entropy, scale_achieved_at, height_achieved_at))]
    
    return angle_entropies, min_angle_entropy, angle_achieved_at, scale_achieved_at, height_achieved_at


#full experiment to take in patient xray and mri volume and register them given the search parameters
def registration_experiment(patient, side, rot_start, rot_num_images, rot_step,scalar, num_bins, num_up, num_down, trans_step, num_inflate, num_deflate, xray_patella_centre, tibial_coordinates, kernel_size, angle, resize =0.5, measure = 'entropy', noise_constant =10, gradient_descent = False):
    start = time.time()
    _, volume = loadPatientMask(patient, side, "1")
    xray_patella_mask = cv2.imread("xrays\\"+patient+"_"+side.lower()+"_xray_contour.png", cv2.IMREAD_GRAYSCALE)
    original_xray = cv2.imread("xrays\\"+patient+"_"+side.lower()+"_xray.jpg", cv2.IMREAD_GRAYSCALE)
    resized_first_xray = cv2.resize(original_xray, (0,0), fx=resize, fy =resize)  #to be saved to show base case 


 
    xray = pre_process_xray_tibialvals(original_xray, xray_patella_mask, xray_patella_centre, tibial_coordinates, kernel_size)
    xray = cv2.resize(xray, (0,0), fx=resize, fy =resize)
    xray_patella_mask =cv2.resize(xray_patella_mask, (0,0), fx=resize, fy =resize)
    np.save("results\\"+patient+"_"+side+"_enhanced_xray.npy", xray)
    
    entropies, min_entropy, angle_achieved_at, scale_achieved_at, height_achieved_at = rot_scale_vert(volume, rot_start, rot_num_images,rot_step ,xray, xray_patella_mask, scalar, num_bins, num_up, num_down, trans_step, num_inflate, num_deflate, resize, measure=measure, noise_constant=noise_constant, gradient_descent=gradient_descent)
    
    end = time.time()
    time_taken = end-start

    print(entropies)
    print("Min entropy was ", min_entropy, " achieved at (angle, scale, height) ", angle_achieved_at, scale_achieved_at, height_achieved_at)
    np.savez_compressed("results\\"+patient+"\\"+measure+"\\angleEntropies.npz", e= entropies, s = [min_entropy, angle_achieved_at, scale_achieved_at, height_achieved_at])
    


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
    
    
    rotation_relative_to_true_coronal = (angle+angle_achieved_at)%360
    if rotation_relative_to_true_coronal >180:
        rotation_relative_to_true_coronal = -1*(360-rotation_relative_to_true_coronal)
    
    
    fg, ax = plt.subplots(2,2)
    fg.suptitle("Viewing Angle: " + str(int(rotation_relative_to_true_coronal)) +", Minimising Angle: " + str(int(angle_achieved_at)) +  ",\n Scale: " + str(round(scale_achieved_at,2))+ ", Height " + str(round(height_achieved_at,2)))
    ax[0][0].imshow(cropped_xray, cmap= 'gray')
    ax[0][0].set_title("Processed X-ray")
    ax[0][0].set_axis_off()
    ax[0][1].imshow(pxray, cmap= 'gray')
    ax[0][1].set_title("Pseudo-xray")
    ax[0][1].set_axis_off()
    ax[1][0].imshow(overlay, cmap = 'gray')
    ax[1][0].set_title("Overlayed Images")
    ax[1][0].set_axis_off()
    ax[1][1].imshow(difference, cmap = 'gray')
    ax[1][1].set_title("Difference Image")
    ax[1][1].set_axis_off()
    #plt.show()

    path = "results\\" + patient + "\\" + measure + "\\"
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.savefig(path+"registered_images.png")
    cv2.imwrite(path + "cropped_xray.png", cropped_xray)
    cv2.imwrite(path + "pxray.png", pxray)
    cv2.imwrite(path + "overlay.png", overlay)
    cv2.imwrite(path + "difference.png", difference)

    

    fg2, ax2 = plt.subplots(1,3)
    ax2[0].imshow(resized_first_xray, cmap ='gray', vmin =0, vmax =255)
    ax2[0].set_title("Original X-ray")
    ax2[0].set_axis_off()
    ax2[1].imshow(unrotated_pxray, cmap ='gray', vmin =0, vmax =255)
    ax2[1].set_title("Unrotated Pseudo-xray")
    ax2[1].set_axis_off()
    ax2[2].imshow(true_xray_overlay, cmap ='gray', vmin = 0, vmax = 255)
    ax2[2].set_title("Registered Pseudo-xray and Xray")
    ax2[2].set_axis_off()
    #plt.savefig("results\\"+patient+"_"+side+"_unprocessed_images.png")
    plt.savefig(path+"unprocessed_images.png")
    cv2.imwrite(path + "original_xray.png", resized_first_xray)
    cv2.imwrite(path + "unrotated_pxray.png", unrotated_pxray)
    cv2.imwrite(path + "overlay.png", overlay)
    cv2.imwrite(path + "proportional_overlay.png", true_xray_overlay)

    return rotation_relative_to_true_coronal, angle_achieved_at,  scale_achieved_at, height_achieved_at, min_entropy, time_taken 
    

def write_record_to_csv(experiment_type, similarity_measure,patient,rotation_relative_to_true_coronal, angle_achieved_at,  scale_achieved_at, height_achieved_at, min_entropy, time_taken ):
    thisrecord = {}
    thisrecord["Patient"] = patient
    thisrecord["Determined Viewing Angle"] = rotation_relative_to_true_coronal
    thisrecord["Minimising Angle"] = angle_achieved_at
    thisrecord["Minimising Scale Factor"] = scale_achieved_at
    thisrecord["Minimising Translation Factor"] = height_achieved_at
    thisrecord["Miniminal Entropy"] = min_entropy
    thisrecord["Time"] = time_taken

    with open("csvs//"+similarity_measure + "_"+ experiment_type+"_results.csv", "a", newline = '') as csvfile:
        fieldnames = ["Patient","Determined Viewing Angle","Minimising Angle","Minimising Scale Factor","Minimising Translation Factor","Miniminal Entropy","Time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(thisrecord)


#for the search procedure we first check angles in steps of 4, doing the registration experiment with parameters 336,11,4, 1, 64, 3, 3, 4, 2, 2,
#then we refine to do 1 degree checks around the best fit


#we will define two experiment types - one which does coars 4-angle steps and one which refines given a rough registration
def coarse_experiment(patient_index, similarity_measure, patients, patient_pat_centres, patient_tib_centres, patient_angles):
    with open("csvs//"+similarity_measure + "_coarse_results.csv", "w", newline = '') as csvfile:
            fieldnames = ["Patient","Determined Viewing Angle","Minimising Angle","Minimising Scale Factor","Minimising Translation Factor","Miniminal Entropy","Time"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    f = open("temp_angles.text", "w")
    f.write(" ")
    f.close()
    for i in patient_index:
        print(i)
        patient = patients[i]
        rotation_relative_to_true_coronal, angle_achieved_at,  scale_achieved_at, height_achieved_at, min_entropy, time_taken= registration_experiment(patients[i], "LEFT", 336,11, 4,1, 64, 3, 3, 4, 2, 2, patient_pat_centres[i], patient_tib_centres[i], 32, patient_angles[i], measure = similarity_measure, noise_constant = 255)
        #registration_experiment(patients[i], "LEFT", 336,22, 2,1, 64, 3, 3, 4, 2, 2, patient_pat_centres[i], patient_tib_centres[i], 32, patient_angles[i], measure = similarity_measure, noise_constant = 255)
        
        write_record_to_csv("coarse", similarity_measure,patient, rotation_relative_to_true_coronal, angle_achieved_at,  scale_achieved_at, height_achieved_at, min_entropy, time_taken)

def fine_experiment(patient_index, similarity_measure, raw_coarse_angles, patients, patient_pat_centres, patient_tib_centres, patient_angles, experiment_type ="fine"):
    with open("csvs//"+similarity_measure + "_"+experiment_type+"_results.csv", "w", newline = '') as csvfile:
        fieldnames = ["Patient","Determined Viewing Angle","Minimising Angle","Minimising Scale Factor","Minimising Translation Factor","Miniminal Entropy","Time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    f = open("temp_angles.text", "w")
    f.write(" ")
    f.close()
    for i in patient_index:
        print(i)
        patient = patients[i]
        rotation_relative_to_true_coronal, angle_achieved_at,  scale_achieved_at, height_achieved_at, min_entropy, time_taken=registration_experiment(patients[i], "LEFT", raw_coarse_angles[i]-3,8, 1,1, 64, 12, 12, 2, 3, 3, patient_pat_centres[i], patient_tib_centres[i], 32, patient_angles[i], measure = similarity_measure, noise_constant = 255)
        write_record_to_csv(experiment_type, similarity_measure, patient, rotation_relative_to_true_coronal, angle_achieved_at,  scale_achieved_at, height_achieved_at, min_entropy, time_taken)

def run_multiple_experiments(patient_index, similarity_measure, experiment_type, patients, patient_pat_centres, patient_tib_centres, patient_angles):

    if experiment_type == "coarse":
        coarse_experiment(patient_index, similarity_measure, patients, patient_pat_centres, patient_tib_centres, patient_angles)
    elif experiment_type =="fine":
        if similarity_measure == 'gmi':
            #gmi_step_4_results_corrected = [4, 10, 356, 10, 355,15,12,358,12,354,5,359,0,345,353,4,2,0,3,16,12,358,11,20,7,356,359,2,11,3] #these are indexed in roder of patients with none missing
            gmi_step_4_results_raw = [0,356,348,0,348,8,4,356,4,344,4,356,356,336,340,0,0,352,0,8,356,352,4,12,4,352,352,4,4,4]
            """for i in range(0,len(gmi_step_4_results_corrected)):
                patient = patients[i]
                raw_angle = (gmi_step_4_results_corrected[i] - patient_angles[i]) %360
                gmi_step_4_results_raw.append(raw_angle)
                f = open("temp_angles.text", "a")
                f.write(str(raw_angle)+",")
                f.close()
            """
            fine_experiment(patient_index, similarity_measure, gmi_step_4_results_raw, patients, patient_pat_centres, patient_tib_centres, patient_angles)
        
        elif similarity_measure == 'entropy':
            entropy_step_4_results_raw = [0,356,352,0,344,4,4,356,0,352,0,356,0,340,344,356,356,352,4,8,356,348,8,352,356,344,356,4,4,8]

            fine_experiment(patient_index, similarity_measure, entropy_step_4_results_raw, patients, patient_pat_centres, patient_tib_centres, patient_angles)

        elif similarity_measure =="mutual_information":
            mi_step_4_results_raw = [356,356,348,0,348,8,4,356,4,352,4,352, 356,336,340,0,356,352,0, 0,356,348,4,8,4,356,356,0,0,0]

            fine_experiment(patient_index, similarity_measure, mi_step_4_results_raw, patients, patient_pat_centres, patient_tib_centres, patient_angles)
        elif similarity_measure == "gmi_e":
            gmie_step_4_results_raw = [0,356,348,0,348,4,4,356,4,344,4,352,356,336,340,0,0,352,0,0,356,352,4,8,4,0,356,4,0,0]
            gmie_step_2_results_raw = [358,356,348,0,348,6,4,358,4,342,4,352,356,336,342,0,358,354,0,2,356,352,4,8,2,0,354,4,2,2]
            
            fine_experiment(patient_index, similarity_measure, gmie_step_4_results_raw, patients, patient_pat_centres, patient_tib_centres, patient_angles)
        
        elif similarity_measure == "gradient_difference":
            gradientdiff_step_4_results_raw = [352,352,356,12,348,336,4,0,0,348,0,4,4,16,340,352,16,352,8,8,356,0,12,16,4,336,348,0,344,356]
            fine_experiment(patient_index, similarity_measure, gradientdiff_step_4_results_raw, patients, patient_pat_centres, patient_tib_centres, patient_angles)
            #will come back to this but will be great to get gradient difference stats too
    
    elif experiment_type =="from_manual":
        manual_raw_angles = []
        j=0
        for i in patient_index:
            while j<i:
                manual_raw_angles.append(None)
                j+=1
            
            patient = patients[i]
            manual_xray_angle = xray_manual_angles[patient]
            manual_raw_angles.append(manual_xray_angle)
            j+=1
        print(manual_raw_angles)
        fine_experiment(patient_index, similarity_measure, manual_raw_angles, patients, patient_pat_centres, patient_tib_centres, patient_angles, experiment_type=experiment_type)
        


def grad_desc_experiments(patients, patient_angles, patient_pat_centres, patient_tib_centres, similarity_measure, initial_step, max_images, decay_constant):
    with open("csvs//"+similarity_measure + "_graddesc_results.csv", "w", newline = '') as csvfile:
            fieldnames = ["Patient","Determined Viewing Angle","Minimising Angle","Minimising Scale Factor","Minimising Translation Factor","Miniminal Entropy","Time"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    f = open("temp_angles.text", "w")
    f.write(" ")
    f.close()
    for i in range (0, len(patients)):
        print(i)
        patient = patients[i]
        #finer search:
        #rotation_relative_to_true_coronal, angle_achieved_at,  scale_achieved_at, height_achieved_at, min_entropy, time_taken = registration_experiment(patient, "LEFT", 0, max_images, initial_step, 1, 64, 12, 12, 2, 3, 3, patient_pat_centres[i], patient_tib_centres[i], 32, patient_angles[i], measure = similarity_measure, noise_constant = 255, gradient_descent = True)
        #coarser search:
        rotation_relative_to_true_coronal, angle_achieved_at,  scale_achieved_at, height_achieved_at, min_entropy, time_taken = registration_experiment(patient, "LEFT", 0, max_images, initial_step, 1, 64, 3, 3, 4, 2, 2, patient_pat_centres[i], patient_tib_centres[i], 32, patient_angles[i], measure = similarity_measure, noise_constant = 255, gradient_descent = True)
        write_record_to_csv("graddesc", similarity_measure,patient, rotation_relative_to_true_coronal, angle_achieved_at,  scale_achieved_at, height_achieved_at, min_entropy, time_taken)
    



def old_order_to_new(old_order, patients):
    old_order_dict = {}
    for i in range(0, len(patients)):
        patient = patients[i]
        old_order_dict[patient] = old_order[i]
    new_order = []
    for i in range (0, len(patients)):
        patient = patients_neworder[i]
        new_order.append(old_order_dict[patient])
    return new_order  


def initialise_time_csv(similarity_measure):
    with open("csvs//time_"+similarity_measure + ".csv", "w", newline = '') as csvfile:
        fieldnames = ["Number","Time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

#************************************************************DATA FOR USE IN TESTING***********************************

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

patients_neworder = ["9002316", "9002411", "9002817", "9030925", "9031141", "9031930", "9031961", "9033937", "9034451", "9034677", "9034812", "9034963", "9911221", "9911721", "9917307", "9918802", "9921811", "9924274",  "9938236", "9943227","9947240", "9958234", "9964731", "9986207", "9986355", "9986838", "9989352", "9989700", "9990192", "9990355"]
patients_oldorder = ["9002316", "9002411", "9002817", "9911221", "9911721", "9917307", "9918802", "9921811", "9924274", "9947240", "9938236", "9943227", "9958234", "9964731", "9986355", "9986838", "9989352", "9989700", "9990192", "9990355", "9986207","9030925", "9031141", "9031930", "9031961", "9033937", "9034451", "9034677", "9034812", "9034963"]
patients = patients_oldorder
patient_tib_centres_dict = {"9002316":(170,300), "9002411":(170,300),"9002817":(170, 270), "9911221":(170,303),"9911721":(170,239), "9917307":(185,325), "9918802":(121,286), "9921811":(179,321),"9924274":(207,312) , "9947240":(170,339), "9938236":(181, 317), "9943227":(203,326), "9958234":(155,343),"9964731":(183,269), "9986355":(200,310), "9986838":(158,307), "9989352":(192,298), "9989700":(170,300), "9990192":(170,311), "9990355":(179,338), "9986207":(212,311), "9030925":(192,335), "9031141":(150,266), "9031930":(120,339), "9031961":(164,332),  "9033937":(178,350), "9034451":(145,296),"9034677":(156,256), "9034812":(159,317), "9034963":(209,329)}
patient_pat_centres_dict = {"9002316":(170,220), "9002411":(170,196), "9002817":(170,186), "9911221":(155,232),"9911721":(182,169), "9917307":(170,200), "9918802":(103,195), "9921811":(178,226), "9924274":(199, 208), "9947240":(169,238), "9938236":(174, 236), "9943227":(206,240), "9958234":(146,239), "9964731":(209,172), "9986355":(211,216), "9986838":(153,198), "9989352":(193,208), "9989700":(178,206), "9990192":(164,222), "9990355":(170,201), "9986207":(197,222), "9030925":(204,158), "9031141":(138,160), "9031930":(100,166), "9031961":(156,235),  "9033937":(170,197), "9034451":(148,199), "9034677":(151,167), "9034812":(153,232), "9034963":(209,238)}
patient_tib_centres = []
patient_pat_centres = []
patient_angles = []
for i in range (0, len(patients)):
    patient = patients[i]
    patient_angles.append(angles[patient+"_LEFT_1"])
    patient_tib_centres.append(patient_tib_centres_dict[patient])
    patient_pat_centres.append(patient_pat_centres_dict[patient])

xray_manual_angles = {"9002316":0, "9002411":356, "9002817":348, "9964731":333, "9030925":352, "9031141": 5, "9031930": 16, "9031961":3, "9033937":355, "9034451":355,"9034677":2, "9034812":0,"9034963":2}


#PATIENT 9964731 is a very good example of an xray with a wildly misaligned patella




"""
#ARCHIVED TEXT:
gmi_step_4_results_corrected = [4, 10, 356, 10, 355,15,12,358,12,354,5,359,0,345,353,4,2,0,3,16,12,358,11,20,7,356,359,2,11,3] #these are indexed in roder of patients with none missing
gmi_step_4_results_raw = []

with open("temp_angles.text", "r") as file:
    a = [line.strip() for line in file]

entropy_step_4_results_raw = list(map(int,a[0].split(",")))
#print(entropy_step_4_results_raw)


for i in range(0,len(gmi_step_4_results_corrected)):#range(len(patients)):
    patient = patients[i]
    raw_angle = (gmi_step_4_results_corrected[i] - patient_angles[i]) %360
    gmi_step_4_results_raw.append(raw_angle)

mi_step_4_results_raw = [356,356,348,0,348,8,4,356,4,352,4,352, 356,336,340,0,356,352,0, 0,356,348,4,8,4,356,356,0,0,0]
"""



patient_index = range(0, len(patients))
similarity_measure = "mutual_information" #can be gmi_e, gmi , entropy
experiment_type = "coarse" #can be fine
run_multiple_experiments(patient_index, similarity_measure, experiment_type, patients, patient_pat_centres, patient_tib_centres, patient_angles)



#grad_desc_experiments(patients, patient_angles, patient_pat_centres, patient_tib_centres, "gmi_e",10, 10, 0.7)
#run_multiple_experiments(patient_index, similarity_measure, experiment_type, patients, patient_pat_centres, patient_tib_centres, patient_angles)


#Code for fine search of full search space:
#i = patients.index("9002817")
#registration_experiment(patients[i], "LEFT", 336,44, 1,1, 64, 12, 12, 2, 3, 3, patient_pat_centres[i], patient_tib_centres[i], 32, patient_angles[i], measure = similarity_measure, noise_constant = 255)