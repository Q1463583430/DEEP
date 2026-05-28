import numpy as np
import h5py
from reload import load_data


train_x,train_y,test_x,test_y,classes=load_data()

#拉平，4d变2d，
train_x = train_x.reshape(train_x.shape[0],-1).T
test_x = test_x.reshape(test_x.shape[0],-1).T

train_x = train_x/255
test_x= test_x/255

#激活函数
def sigmoid(z):
    a=1/(1+np.exp(-z))
    return a
#z=w.T*X+b，先初始出w，b，然后得到z，得到a，然后计算cost，再接着计算dw，db
w = np.zeros((train_x.shape[0],1))
b=0.0
assert(w.shape == (train_x.shape[0], 1))
assert(isinstance(b,float) or isinstance(b,int))

#样本数量,计算cost，和dw，db
def cost_and_grad(w,b,train_x,train_y):  
    z = np.dot(w.T,train_x)+b
    s = sigmoid(z)
    m = train_x.shape[1]
    cost = -1/m *np.sum(train_y*np.log(s)+(1-train_y)*np.log(1-s))
    dw = 1/m* np.dot(train_x,(s-train_y).T)
    db = 1/m * np.sum(s-train_y)
    return dw,db,cost
#迭代次数numit=1000，学习效率阿尔法rate=0.05

costs = []
numit = 1000
rate = 0.05
for i in range(numit):
    dw,db,cost = cost_and_grad(w,b,train_x,train_y)
    w = w - dw * rate
    b = b - db * rate
    if i % 100 == 0:
        costs.append(cost)
    if i % 100 == 0:
        print("损失结果为%i:%f"%(i, cost))

def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i] <=0.5:
            Y_prediction[0,i] = 0 
        else:
            Y_prediction[0,i] = 1
    return Y_prediction


Y_prediction_test = predict(w,b,test_x)
Y_prediction_train = predict(w,b,train_x)
print("训练集准确率: {} ".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
print("测试集准确率: {} ".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))
     


