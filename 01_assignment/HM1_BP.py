import numpy as np
from utils import read_img


if __name__ == "__main__":

    #input
    input_vector = np.zeros((10,784)) #10张图的特征向量
    for i in range(10):
        input_vector[i,:] = read_img("mnist_subset/"+str(i)+".png").reshape(-1)/255.
    gt_y = np.zeros((10,1)) #ground truth 
    gt_y[0] =1  #第一个是0 别的都不是0

    np.random.seed(14)

    #Intialization MLP  (784 -> 16 -> 1)
    MLP_layer_1 = np.random.randn(784,16) #weight
    MLP_layer_2 = np.random.randn(16,1)
    lr=1e-1 #learning rate
    loss_list=[]

    for i in range(50):
        #Forward
        output_layer_1 = input_vector.dot(MLP_layer_1) #（10，16）z1
        output_layer_1_act = 1 / (1+np.exp(-output_layer_1))  #sigmoid activation function （10，16）a1 
        output_layer_2 = output_layer_1_act.dot(MLP_layer_2) #（10，1）z2
        pred_y = 1 / (1+np.exp(-output_layer_2))  #sigmoid activation function （10，1）预测每个图片是0的概率 a2
        loss = -( gt_y * np.log(pred_y) + (1-gt_y) * np.log(1-pred_y)).sum() #cross-entroy loss.和ground truth比较 求出loss
        print("iteration: %d, loss: %f" % (i+1 ,loss))
        loss_list.append(loss)


        # Backward : compute the gradient of paratmerters of layer1 (grad_layer_1) and layer2 (grad_layer_2)

        delta_2 = pred_y - gt_y  # delta_2 = dL/dz2 which is a2-y (10,1)
        grad_layer_2 = output_layer_1_act.T @ delta_2 # grad_layer_2 = dL/dw2  (16,10)(10,1) = (16,1)

        delta_1 = delta_2 @ MLP_layer_2.T * output_layer_1_act * (1 - output_layer_1_act) # delta_1 = dL/dz1 sigmoid求导是sigmoid(x)*(1-sigmoid(x))
        grad_layer_1 = input_vector.T @ delta_1 # grad_layer_1 = dL/dw1

        MLP_layer_1 -= lr * grad_layer_1 #需要优化的参数 两层的W
        MLP_layer_2 -= lr * grad_layer_2

    np.savetxt("result/HM1_BP.txt", loss_list)