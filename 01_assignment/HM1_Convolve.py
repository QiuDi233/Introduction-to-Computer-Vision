import numpy as np
from scipy.linalg import toeplitz
from scipy import signal
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    if type=="zeroPadding":
        h, w = img.shape
        padding_img = np.zeros((h + padding_size*2, w + padding_size*2))
        padding_img[padding_size : padding_size + h, padding_size : padding_size + w] = img
        return padding_img
    elif type=="replicatePadding":
        h, w = img.shape
        padding_img = np.zeros((h + padding_size*2, w + padding_size*2))
        padding_img[padding_size : padding_size + h, padding_size : padding_size + w] = img
        #填充四个角
        padding_img[:padding_size, : padding_size] = img[0][0]
        padding_img[:padding_size, w + padding_size : w + padding_size*2] = img[0][w-1]
        padding_img[h + padding_size : h + padding_size*2, : padding_size] = img[h-1][0]
        padding_img[h + padding_size : h + padding_size*2,w + padding_size : w + padding_size*2] = img[h-1][w-1]
        #填充四个边
        padding_img[ : padding_size, padding_size : padding_size + w] = img[:1,:]
        padding_img[h + padding_size : , padding_size : padding_size + w] = img[-1:,:]
        padding_img[padding_size : h + padding_size,  : padding_size] = img[:,:1]
        padding_img[padding_size : h + padding_size, w + padding_size : w + padding_size*2] = img[:,-1:]
        
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        这个函数实现时用的翻转kernel
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    padding_img = padding(img,1,"zeroPadding") #padding为1时，卷后size不变 padding_img是8*8

    #build the Toeplitz matrix and compute convolution
    I = img
    F = kernel

    # number columns and rows of the input 
    I_row_num, I_col_num = I.shape 

    # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    output_row_num = I_row_num + F_row_num - 1
    output_col_num = I_col_num + F_col_num - 1
    #print('output dimension:', output_row_num, output_col_num)

    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                            (0, output_col_num - F_col_num)),
                            'constant', constant_values=0)
    #print('F_zero_padded: ', F_zero_padded)

    from scipy.linalg import toeplitz

    # use each row of the zero-padded F to creat a toeplitz matrix. 
    #  Number of columns in this matrices are same as numbe of columns of input signal
    toeplitz_list = []
    for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
        c = F_zero_padded[i, :] # i th row of the F 
        r = np.r_[c[0], np.zeros(I_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                            # the result is wrong
        toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)
        #print('F '+ str(i)+'\n', toeplitz_m)

    # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    #print('doubly indices \n', doubly_indices)


    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

    #print('doubly_blocked: ', doubly_blocked)
    
    def matrix_to_vector(input):
        input_h, input_w = input.shape
        output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
        # flip the input matrix up-down because last row should go first
        input = np.flipud(input) 
        for i,row in enumerate(input):
            st = i*input_w
            nd = st + input_w
            output_vector[st:nd] = row
        return output_vector

    # call the function
    vectorized_I = matrix_to_vector(I)
    #print('vectorized_I: ', vectorized_I)

    # get result of the convolution by matrix mupltiplication
    result_vector = np.matmul(doubly_blocked, vectorized_I)
    #print('result_vector: ', result_vector)
            
    def vector_to_matrix(input, output_shape):
        output_h, output_w = output_shape
        output = np.zeros(output_shape, dtype=input.dtype)
        for i in range(output_h):
            st = i*output_w
            nd = st + output_w
            output[i, :] = input[st:nd]
        # flip the output matrix up-down to get correct result
        output=np.flipud(output)
        return output
    
    # reshape the raw rsult to desired matrix form
    out_shape = [output_row_num, output_col_num]
    my_output = vector_to_matrix(result_vector, out_shape) #8x8

    #print('Result of implemented method: \n', my_output)

    
    #lib_output = signal.convolve2d(padding_img, F, "valid")
    #np.savetxt("result/lib_result.txt", lib_output)

    output = my_output[1:-1,1:-1] #6x6
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here
    h, w = img.shape
    k = kernel.shape[0]
    #构造一个(h-k+1)*(w-k+1) x k^2的矩阵 和把kernel拉成k^2 x 1的向量 最后展平成(h-k+1)x(w-k+1)的结果
    kernel_flat = kernel.flatten() 

    output_h = h-k+1
    output_w = w-k+1
    patch_matrix = np.zeros((output_h*output_w,k**2))
    #每一行都是一个kxk的位置的展开 按kernel会卷到的位置把它填上
    #for i, j in np.ndindex((output_h, output_w)):
    #   patch_matrix[i * output_w + j] = img[i:i+k, j:j+k].ravel()

    rows, cols = np.meshgrid(np.arange(output_h),np.arange(output_w),indexing='ij')
    a,b = np.meshgrid(np.arange(k),np.arange(k),indexing='ij')

    patch_matrix = img[rows[:,:,np.newaxis,np.newaxis] + a,cols[:,:,np.newaxis,np.newaxis] + b]
    patch_matrix = patch_matrix.reshape(output_h*output_w,-1)
    
    output_flat = patch_matrix@kernel_flat
    output = output_flat.reshape(output_h, output_w)

   

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #lib_output = signal.convolve2d(input_array, input_kernel, "valid")
    #np.savetxt("result/lib_result2.txt", lib_output)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)



    print("end")
    