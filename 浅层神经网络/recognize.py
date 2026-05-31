import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from reload_data import load_random_data


#第一步加载数据
#第二步 初始化表格和w1，b1，w2，b2
#第三步 正向传播，a1是tanh(z1),a2是sigmoid(z2)和cost
#第四步 计算梯度
#第五步 反向传播，更新w1，b1，w2，b2
#得到所有结果，跟原本结果比较，得出预测率

X,Y=load_random_data()

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s
    
def Init(X,Y):
    np.random.seed(2)
    w1=np.random.randn(4,X.shape[0])*0.01
    b1=np.zeros((4,1))
    w2=np.random.randn(Y.shape[0],4)*0.01
    b2=np.zeros((Y.shape[0],1))
    return w1,w2,b1,b2
    
def cauluate_cost(A2,Y):
    m=Y.shape[1]
    cost=-1/m*np.sum((np.dot(Y,np.log(A2).T)+np.dot(1-Y,np.log(1-A2).T)))
    cost = np.squeeze(cost)
    return cost

def Grad(w1,w2,A1,A2,X,Y):
    #反向传播，先计算最后一层的梯度，再往前计算
    m=X.shape[1]
    dz2=A2-Y
    dw2=1/m*np.dot(dz2,A1.T)
    db2=1/m*np.sum(dz2,axis=1,keepdims=True)
    dz1=np.dot(w2.T,dz2)*(1-np.power(A1,2))
    dw1=1/m*np.dot(dz1,X.T)
    db1=1/m*np.sum(dz1,axis=1,keepdims=True)

    return dw1,db1,dw2,db2

def update_grad(w1,w2,b1,b2,dw1,dw2,db1,db2,learning_rate):
    w1=w1-learning_rate*dw1
    w2=w2-learning_rate*dw2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2
    return w1,w2,b1,b2

def model(X,Y,learning_rate=0.005,num=10000):
    w1,w2,b1,b2=Init(X,Y)

    for i in range(0,num):
        z1=np.dot(w1,X)+b1
        A1=np.tanh(z1)
        z2=np.dot(w2,A1)+b2
        A2=sigmoid(z2)
        cost_now=cauluate_cost(A2,Y)
        dw1,db1,dw2,db2=Grad(w1,w2,A1,A2,X,Y)
        w1,w2,b1,b2=update_grad(w1,w2,b1,b2,dw1,dw2,db1,db2,learning_rate)
        if i%1000==0:
            print("迭代次数 %i, cost为 %f" % (i, cost_now))
    return w1,w2,b1,b2

w1,w2,b1,b2=model(X,Y,learning_rate=0.005,num=10000)
z1=np.dot(w1,X)+b1
A1=np.tanh(z1)
z2=np.dot(w2,A1)+b2
A2=sigmoid(z2)
predictions=(A2>0.45).astype(int)
accuracy=np.mean(Y == predictions)*100
print('准确率： %.2f%%' % accuracy)





        







