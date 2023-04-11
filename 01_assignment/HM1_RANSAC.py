import numpy as np
from utils import draw_save_plane_with_points


def get_plane(points):
    p1 = points[:, 0]
    p2 = points[:, 1]
    p3 = points[:, 2]
    a = ( (p2[:,1]-p1[:,1])*(p3[:,2]-p1[:,2])-(p2[:,2]-p1[:,2])*(p3[:,1]-p1[:,1]) )
    b = ( (p2[:,2]-p1[:,2])*(p3[:,0]-p1[:,0])-(p2[:,0]-p1[:,0])*(p3[:,2]-p1[:,2]) )
    c = ( (p2[:,0]-p1[:,0])*(p3[:,1]-p1[:,1])-(p2[:,1]-p1[:,1])*(p3[:,0]-p1[:,0]) )
    d = ( 0-(a*p1[:,0]+b*p1[:,1]+c*p1[:,2]) )
    return np.stack([a, b, c, d], axis=-1)


def is_inliers(p,para,threshold): #para (sample_time, 4) 该函数表示一个点对于这sample_time种情况来说是否是内点 返回(sample_time,)大小
    A = para[:,0] #(sample_time,)
    B = para[:,1]
    C = para[:,2]
    D = para[:,3]
    num1 = A*p[0] + B*p[1] + C*p[2] + D #(sample_time,)
    num2 = np.sqrt(np.sum(np.square([A, B, C]),axis=0))  #(sample_time,)
    dis = np.abs(num1)/num2  #(sample_time,1)
    judge_inliers = np.zeros_like(dis) #(sample_time,)
    judge_inliers[dis<threshold] = 1 #(sample_time,)
    return judge_inliers #(sample_time,1)


if __name__ == "__main__":


    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt") #(130,3)
    

    #RANSAC
    # we recommend you to formulate the palnace function as:  A*x+B*y+C*z+D=0    
    #抽1次不含outlier的概率p=C(3,100)/C(3,130)   1-（1-p)^n>0.999 解出来大概11.几 这里保证概率 向上取到15 反正向量化跑起来很快
    sample_time = 15 #more than 99.9% probability at least one hypothesis does not contain any outliers 
    distance_threshold = 0.05

    # sample points group
    # 生成随机的points group 每组3个 因为3点确定一个平面
    # 抽sample_time组，每组抽3个 
    selected = np.random.choice(np.arange(noise_points.shape[0]), size=(sample_time,3), replace=False) #(sample_time,3)
    selected_points = noise_points[selected] #(sample_time,3,3)
    
    # estimate the plane with sampled points group  
    # 用选择的点集来生成平面
    planes_parameter = get_plane(selected_points) #（sample_time，4） 每组的4个ABCD参数 一共sample_time组
    cnt = np.zeros(sample_time) #每组有多少个内点

    #evaluate inliers (with point-to-plance distance < distance_threshold)
    #计算生成的平面的内点数量
    #获得对每组来说 130个点是否是内点 是为1 不是为0
    result = np.apply_along_axis(is_inliers, 1, noise_points, planes_parameter, distance_threshold) #(130,st）
    cnt = np.sum(result,axis = 0) #(st,)
        
    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    # 用内点最多的一组来作为参数 依据该组的局内点用最小二乘法拟合平面 
    max_idx = np.argmax(cnt)
    #最好的一组的内点
    optimal_points_idx = result[:,max_idx] #(130,) 0 or 1 代表是否是内点
    optimal_points = noise_points[optimal_points_idx==1] #(103,3)这样子

    x = optimal_points[:,0]
    y = optimal_points[:,1]
    z = optimal_points[:,2]
    #ax+by+d=z
    data = np.column_stack((x, y, np.ones_like(x)))
    result_para, _, _, _ = np.linalg.lstsq(data, z, rcond=None)
    optimal_A = result_para[0]
    optimal_B = result_para[1]
    optimal_C = -1
    optimal_D = result_para[2]
    pf = [optimal_A, optimal_B, optimal_C, optimal_D]
    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)

