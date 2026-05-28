import numpy as np
import h5py


def load_data():
    train_dataset = h5py.File('mimidata/train_catvnoncat.h5',"r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])#train特征
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])#train标签

    test_dataset = h5py.File('mimidata/test_catvnoncat.h5',"r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #test特征
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) #test标签
    

    classes = np.array(test_dataset["list_classes"][:])
    #转为二维数组
    train_set_y_orig = train_set_y_orig.reshape(1,train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(1,test_set_y_orig.shape[0])
    #把(209,)转换为(1,209)的二维数组，转置之后乘积
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes

