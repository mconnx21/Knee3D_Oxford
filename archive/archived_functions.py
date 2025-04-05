

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
"""


    #old implementation of entrtopy below but good for demonstrating what is actually going on here

    def entropy:
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






