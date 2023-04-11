import numpy as np
from utils import  read_img, draw_corner
from HM1_Convolve import convolve, Sobel_filter_x,Sobel_filter_y,padding



def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: list
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get 
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for details of corner_response_function, please refer to the slides.

    corner_list = []
    #计算两个方向的梯度
    x_grad = Sobel_filter_x(input_img)
    y_grad = Sobel_filter_y(input_img)

    xx_matrix = np.multiply(x_grad,x_grad)
    yy_matrix = np.multiply(y_grad,y_grad)
    xy_matrix = np.multiply(x_grad,y_grad)

    h, w = input_img.shape
    #做滤波 这里直接用长方形的 即均值滤波 1 in window 0 outside
    # 不同的局部窗口会得到不同的M矩阵

    #用来在窗口内遍历 window_size=5时 arange是(-2,3) 也就是-2 -1 0 1 2 
    '''
    for i in range(h):
        for j in range(w):
            xx_sum = xx_matrix[i:i+window_size,j:j+window_size].sum() 
            xy_sum = xy_matrix[i:i+window_size,j:j+window_size].sum()
            yy_sum = yy_matrix[i:i+window_size,j:j+window_size].sum()
            M = np.array([[xx_sum, xy_sum],
                          [xy_sum, yy_sum ]])
            #角点响应值R = detM - alphax(traceM)^2  alpha是一个常数 一般在0.04~0.06 这里给的是0.04
            theta = np.linalg.det(M)- alpha*(np.trace(M))**2
            if theta > threshold:
                corner_list.append((i+1,j+1,theta))'''
    
    window_kernel = np.ones((5,5))
    #xx_sum_matrix（i，j）的值表示上面for循环里的xx_sum
    xx_sum_matrix = convolve(padding(xx_matrix,2,"replicatePadding"), window_kernel)
    xy_sum_matrix = convolve(padding(xy_matrix,2,"replicatePadding"), window_kernel)
    yy_sum_matrix = convolve(padding(yy_matrix,2,"replicatePadding"), window_kernel)

    M = np.stack((np.stack((xx_sum_matrix, xy_sum_matrix), axis=-1), np.stack((xy_sum_matrix, yy_sum_matrix), axis=-1)), axis=-2)
    #每个点处的响应值theta
    theta = np.linalg.det(M) - alpha * (np.trace(M, axis1=-2, axis2=-1)**2)

    # 将符合条件的角点加入列表中
    i, j = np.where(theta > threshold)
    corner_list = list(zip(i, j, theta[i,j]))

    return corner_list # the corners in corne_list: a tuple of (index of rows, index of cols, theta)



if __name__=="__main__":

    #Load the input images
    input_img = read_img("hand_writting.png")/255.

    #you can adjust the parameters to fit your own implementation 
    window_size = 5
    alpha = 0.04
    threshold = 10

    corner_list = corner_response_function(input_img,window_size,alpha,threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key = lambda x: x[2], reverse = True)
    NML_selected = [] 
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted :
        for j in NML_selected :
            if(abs(i[0] - j[0] <= dis) and abs(i[1] - j[1]) <= dis) :
                break
        else :
            NML_selected.append(i[:-1])


    #save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
