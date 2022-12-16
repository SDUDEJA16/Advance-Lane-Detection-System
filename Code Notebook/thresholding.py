import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def extract_yellow(img):
    yellow_mask=np.zeros_like(img[:,:,0])
    yellow_mask[((img[:,:,0]>=15)&(img[:,:,0]<=35))&((img[:,:,1]>=30)&(img[:,:,1]<=204))&((img[:,:,2]>=115)&(img[:,:,2]<=255))]=1
    return yellow_mask

def extract_white(img):
    white_mask=np.zeros_like(img[:,:,0])
    white_mask[((img[:,:,0]>=0)&(img[:,:,0]<=255))&((img[:,:,1]>=200)&(img[:,:,1]<=255))&((img[:,:,2]>=0)&(img[:,:,2]<=255))]=1
    return white_mask

def color_threshold_image(img):
    hls=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    yellow_mask=extract_yellow(hls)
    white_mask=extract_white(hls)
    final_mask=np.zeros_like(hls[:,:,0])
    final_mask[((white_mask==1)|(yellow_mask==1))]=1
    return final_mask   
	
def sobel_thresh_in_xory(gray, orient='x',sobel_kernel=3,thresh=(0, 255)): 
  
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0,sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1,sobel_kernel)
    abs_sobel = np.absolute(sobel)
    converted_sobel=np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(converted_sobel)
    thresh_min=thresh[0]
    thresh_max=thresh[1]
    binary_output[(converted_sobel >= thresh_min) & (converted_sobel <= thresh_max)] = 1
    return binary_output
	
def magnitude_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobel=np.sqrt(sobelx*sobelx+sobely*sobely)
    
    converted_sobel=np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    binary_output = np.zeros_like(converted_sobel)
    thresh_min=thresh[0]
    thresh_max=thresh[1]
    binary_output[(converted_sobel >= thresh_min) & (converted_sobel <= thresh_max)] = 1
    return binary_output	
	
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction=np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(direction)
    binary_output[(direction>=thresh[0]) & (direction<=thresh[1])]=1
    return binary_output

def combine_threshold(best_x_sobel,best_y_sobel,best_xy_sobel,gray,kernel=3,thresh=(0,np.pi/2)):
    dir_mask=dir_threshold(gray,kernel,thresh)
    final_sobel=np.zeros_like(dir_mask)
    final_sobel[(best_x_sobel==1)|((best_y_sobel==1)&(best_xy_sobel==1)&(dir_mask==1))]=1
    return final_sobel
    
def combine_color_and_gradient_threshold(img):
    #image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image=cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:,:,0]
    best_x_sobel=sobel_thresh_in_xory(image,orient='x',sobel_kernel=15,thresh=(20,120))
    best_y_sobel=sobel_thresh_in_xory(image,orient='y',sobel_kernel=15,thresh=(20,120))
    best_xy_sobel=magnitude_thresh(image,sobel_kernel=15,thresh=(80,100))
    combined_sobel=combine_threshold(best_x_sobel,best_y_sobel,best_xy_sobel,image, kernel=15, thresh=(np.pi/4, np.pi/2)) 
    color_mask=color_threshold_image(img)
    final_binary_image=np.zeros_like(color_mask)
    final_binary_image[(combined_sobel==1)|(color_mask==1)]=1
    return final_binary_image
