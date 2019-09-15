import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os


#load images
path='zzh/fshare2/Rotation/zhh/package-v1/sdata7/deeplearning_results/382412-2_1/'
dirs=os.listdir(path)
total=[]
for i in dirs:
    df=pd.read_csv(path+i,sep='\t',header=None)
    total.append(df)
df=pd.concat(total[:],ignore_index=False)
df=df.reset_index(drop=True)
df[1] = df[1].astype('float32')
df[2] = df[2].astype('float32')
df[3] = df[3].astype('float32')

imgs=[]
labels=[]
count=0
count1=0
count2=0
for i in range(df.shape[0]):
    a=max(df[1][i],df[2][i],df[3][i])
    if a==df[1][i]:
        count+=1
        if count<=4000:
            label=[1,0,0]
            img=cv2.imread('zzh'+df[0][i],3)
            imgs.append(img)
            labels.append(label)
    elif a==df[2][i]:
        count1=count1+1
        if count1<=4000:
            label=[0,1,0]
            img=cv2.imread('zzh'+df[0][i],3)
            imgs.append(img)
            labels.append(label)
    else:
        count2=count2+1
        if count2<=4000:
            label=[0,0,1]
            img=cv2.imread('zzh'+df[0][i],3)
            imgs.append(img)
            labels.append(label)

imgs=np.array(imgs)/255
labels=np.array(labels)


#divide images into training set and validation set
ratio=0.8
num_example=imgs.shape[0]
s=np.int(num_example*ratio)
x_train=imgs[:s]
y_train=labels[:s]
x_val=imgs[s:]
y_val=labels[s:]


#build the network
def placeholder(n_H0,n_W0,n_C0,n_y):
    X=tf.placeholder(shape=(None,n_H0,n_W0,n_C0),dtype=tf.float32)
    Y=tf.placeholder(shape=(None,n_y),dtype=tf.float32)
    return X,Y

X,Y=placeholder(80,80,3,3)

def parameters():
    tf.set_random_seed(1)
    W1=tf.get_variable(name='W1',dtype=tf.float32,shape=(5,5,3,32),initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2=tf.get_variable(name='W2',dtype=tf.float32,shape=(5,5,32,64),initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3=tf.get_variable(name='W3',dtype=tf.float32,shape=(3,3,64,128),initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4=tf.get_variable(name='W4',dtype=tf.float32,shape=(10*10*128,512),initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W5=tf.get_variable(name='W5',dtype=tf.float32,shape=(512,3),initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b1=tf.get_variable(name='b1',dtype=tf.float32,shape=(32))
    b2=tf.get_variable(name='b2',dtype=tf.float32,shape=(64))
    b3=tf.get_variable(name='b3',dtype=tf.float32,shape=(128))
    b4=tf.get_variable(name='b4',dtype=tf.float32,shape=(512))
    b5=tf.get_variable(name='b5',dtype=tf.float32,shape=(3))
    parameters={'W1':W1,'W2':W2,'W3':W3,'W4':W4,'W5':W5,'b1':b1,'b2':b2,'b3':b3,'b4':b4,'b5':b5}
    return parameters

def forward_pro(X,parameters):
    W1=parameters['W1']
    W2=parameters['W2']
    W3=parameters['W3']
    W4=parameters['W4']
    W5=parameters['W5']
    b1=parameters['b1']
    b2=parameters['b2']
    b3=parameters['b3']
    b4=parameters['b4']
    b5=parameters['b5']
    Z1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    A1=tf.nn.relu(Z1+b1)
    P1=tf.nn.max_pool(A1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    Z2=tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    A2=tf.nn.relu(Z2+b2)
    P2=tf.nn.max_pool(A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    Z3=tf.nn.conv2d(P2,W3,strides=[1,1,1,1],padding='SAME')
    A3=tf.nn.relu(Z3+b3)
    P3=tf.nn.max_pool(A3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    P3=tf.reshape(P3,[-1,10*10*128])
    h1=tf.nn.relu(tf.matmul(P3,W4)+b4)
    n_drop=tf.nn.dropout(h1,0.6)
    Z3=tf.matmul(n_drop,W5)+b5
    return Z3
 
def loss(Z3,Y):
    los=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
    return los


#shuffle batch
def minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#build the model
def model(X_train,Y_train,X_test,Y_test,num_epoches=60,minibatch_size=100,print_cost=True):
    tf.reset_default_graph()
    tf.set_random_seed(1)
    (m,n_H0,n_W0,n_C0)=X_train.shape
    n_Y=np.array(Y_train.shape[1])
    costs=[]
    X,Y=placeholder(n_H0,n_W0,n_C0,n_Y)
    para=parameters()
    Z3=forward_pro(X,para)
    los=loss(Z3,Y)
    optimizer=tf.train.AdamOptimizer(1e-4).minimize(los)
    predict=tf.argmax(Z3,1)
    correct_prediction=tf.equal(predict,tf.argmax(Y,1))
    acc=tf.reduce_mean(tf.cast(correct_prediction,'float'))
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epoches):
            minibatch_cost=0
            accuracy=0
            batch=1
            for x_train, y_train in minibatches(X_train, Y_train, minibatch_size, shuffle=True):
                _,temp_cost,a=sess.run([optimizer,los,acc],feed_dict={X:x_train,Y:y_train})

                minibatch_cost=minibatch_cost+temp_cost
                cost=minibatch_cost/batch
                accuracy=accuracy+a
                fin_acc=accuracy/batch
                batch+=1
            if print_cost==True and epoch%5==0:
                print('cost after epoch %i:%f'%(epoch,cost))
                print('acc%f'%fin_acc)
            if print_cost==True and epoch%1==0:
                costs.append(cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.show()
        plt.savefig('a')
        predict=tf.argmax(Z3,1)
        correct_prediction=tf.equal(predict,tf.argmax(Y,1))
        acc=tf.reduce_mean(tf.cast(correct_prediction,'float'))
        test_acc=acc.eval({X:X_test,Y:Y_test})
        print('test_acc',test_acc)
        return test_acc,para
    
_,pa=model(x_train,y_train,x_val,y_val)
