import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img
import math
def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)
    direction_grad = np.arctan2(y_grad,x_grad)*(180/np.pi) #arctan2 return [-pi,pi]
    direction_grad[direction_grad<0] += 180 # 范围0-180
    #simply to 4 directions
    direction_grad[(0<=direction_grad) & (direction_grad<22.5)] = 0
    direction_grad[(22.5<=direction_grad) & (direction_grad<67.5)] = 45
    direction_grad[(67.5<=direction_grad) & (direction_grad<112.5)] = 90
    direction_grad[(112.5<=direction_grad) & (direction_grad<157.5)] = 135
    direction_grad[(157.5<=direction_grad) & (direction_grad<=180)] = 0
    
    write_img("result/mag.png", magnitude_grad*255)
    write_img("result/dir.png", direction_grad*255)

    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   
    NMS_output = np.copy(grad_mag)

    NMS_output[(abs(grad_dir - 0) < 1e-6) \
                   & ((np.roll(grad_mag, shift=1, axis=1)>grad_mag) \
                        | (np.roll(grad_mag, shift=-1, axis=1)>grad_mag))] = 0
    #write_img("result/NMS_mag1.png", NMS_output*255)

    NMS_output[(abs(grad_dir - 90) < 1e-6) \
                   & ((np.roll(grad_mag, shift=1, axis=0)>grad_mag )
                        | (np.roll(grad_mag, shift=-1, axis=0)>grad_mag))] = 0
    #write_img("result/NMS_mag2.png", NMS_output*255)
    NMS_output[(abs(grad_dir - 45) < 1e-6) \
                   & ((np.roll(np.roll(grad_mag, shift=1, axis=1), shift=1, axis=0)>grad_mag )\
                        | (np.roll(np.roll(grad_mag, shift=-1, axis=1), shift=-1, axis=0)>grad_mag))] = 0
    #write_img("result/NMS_mag3.png", NMS_output*255)
    NMS_output[(abs(grad_dir - 135) < 1e-6) \
                   & ((np.roll(np.roll(grad_mag, shift=-1, axis=0), shift=1, axis=1)>grad_mag )\
                        | (np.roll(np.roll(grad_mag, shift=1, axis=0), shift=-1, axis=1)>grad_mag))] = 0
    
    #write_img("result/NMS_mag.png", NMS_output*255)

    return NMS_output 
            


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """
    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.10
    high_ratio = 0.30
    
    judge = np.zeros_like(img) #高于max 1 低于min -1 介于两者之间 0
    judge[img>high_ratio] = 1
    judge[img<low_ratio] = -1
    num_cnt = np.count_nonzero(judge) #统计非0的数量 如果不再改变 就退出循环
    num_nxt = np.count_nonzero(judge)
    while True:
        #check每个像素 如果它的judge是0 且与它相邻的像素有judge为1 那么它也设为1
        neighbors = np.roll(np.roll(judge, shift=1, axis=0), shift=1, axis=1) \
                + np.roll(np.roll(judge, shift=1, axis=0), shift=-1, axis=1) \
                + np.roll(np.roll(judge, shift=-1, axis=0), shift=1, axis=1) \
                + np.roll(np.roll(judge, shift=-1, axis=0), shift=-1, axis=1) \
                + np.roll(judge, shift=1, axis=0) \
                + np.roll(judge, shift=-1, axis=0) \
                + np.roll(judge, shift=1, axis=1) \
                + np.roll(judge, shift=-1, axis=1)
        
        judge[(abs(judge - 0)<1e-6) & (neighbors >0)] = 1 

        num_nxt = np.count_nonzero(judge)
        if num_cnt == num_nxt:
            break
        num_cnt = num_nxt

    output = np.where(abs(judge-1)<1e-6,1,0)
    return output 



if __name__=="__main__":

    #Load the input images
    input_img = read_img("lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
