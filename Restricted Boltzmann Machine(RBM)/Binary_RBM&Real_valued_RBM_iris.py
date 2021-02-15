import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

#CD-1 real-valued(input) RBM algorithm
def Gau_Ber_RBM(data, _w,_a,_b, learning_rate, phase='training'):
    if phase=='training':
        h0_p=sigmoid(np.matmul(data,_w)+_b)
        h0=np.random.binomial(1,h0_p)
        x_mu=np.matmul(h0,np.transpose(_w))+_a
        x=np.random.normal(x_mu,1)
        h1_p=sigmoid(np.matmul(x,_w)+_b)
        h1=np.random.binomial(1,h1_p)
        w=_w+learning_rate*(np.matmul(np.transpose(data),h0)-np.matmul(np.transpose(x),h1))/len(data)
        a=_a+learning_rate*(np.mean(data-x,0))
        b=_b+learning_rate*(np.mean(h0-h1,0))
        return w,a,b
    
    elif phase=='loss':
        h0_p=sigmoid(np.matmul(data,_w)+_b)
        h0=np.round(h0_p)
        x=np.matmul(h0, np.transpose(_w))+_a
        reconstruction_error=np.mean((data-x)**2)
        return reconstruction_error
    
    else:
        print('phase must be training or loss')

#CD-1 binary(input) RBM algorithm
def Ber_Ber_RBM(data,_w,_a,_b, learning_rate, phase='training'):
    if phase=='training':
        h0_p=sigmoid(np.matmul(data,_w)+_b)
        h0=np.random.binomial(1,h0_p)
        x_p=sigmoid(np.matmul(h0, np.transpose(_w))+_a)
        x=np.random.binomial(1,x_p)
        h1_p=sigmoid(np.matmul(x,_w)+_b)
        h1=np.random.binomial(1,h1_p)
        w=_w+learning_rate*(np.matmul(np.transpose(data),h0)-np.matmul(np.transpose(x),h1))/len(data)
        a=_a+learning_rate*(np.mean(data-x,0))
        b=_b+learning_rate*(np.mean(h0-h1,0))
        return w,a,b
    
    elif phase=='loss':
        h0_p=sigmoid(np.matmul(data,_w)+_b)
        h0=np.round(h0_p)
        x_p=sigmoid(np.matmul(h0, np.transpose(_w))+_a)
        x=np.round(x_p)
        reconstruction_error=np.mean((data-x)**2)
        return reconstruction_error

    else:
        print('phase must be training or loss')


#data upload
url='https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
iris=pd.read_csv(url)
iris.head()

n=iris.shape[0]
irisX=np.array(iris.iloc[:,:4]) #sepal_length, sepal_width, petal_length, petal_width
irisY=iris.iloc[:,4] #species

#input standardization for real-valued RBM
moments=np.mean(irisX, 0), np.var(irisX,0)
standardized_irisX=(irisX-moments[0])/np.sqrt(moments[1])

#input standardization for binary RBM
minmax=np.amin(irisX,0), np.amax(irisX,0)
normalized_irisX=(irisX-minmax[0])/(minmax[1]-minmax[0])

#input node 4, hidden node 2
_w=np.random.normal(size=[4,2], scale=0.1)
_a=np.zeros([4])
_b=np.zeros([2])
tr_h=sigmoid(np.matmul(normalized_irisX, _w)+_b)

#lr, epoch, minibatch size
learning_rate=5*1e-3
max_epoch=1500
mbs=5

#real valued RBM training
for learning_epoch in range(max_epoch):
    rannum=np.random.permutation(len(standardized_irisX)) #randomly permute a sequence (index)
    num_batch=int(len(standardized_irisX)/mbs)
    for it in range(num_batch):
        batch_X=standardized_irisX[rannum[it*mbs:(it+1)*mbs]] #random index->batch_X
        w,a,b=Gau_Ber_RBM(batch_X,_w,_a,_b, learning_rate, phase='training')
    
    if (learning_epoch+1)%100==0:
        print(Gau_Ber_RBM(standardized_irisX, w,a,b, learning_rate, phase='loss'))

real_h=sigmoid(np.matmul(standardized_irisX, w)+b)

#result scatter
plt.scatter(real_h[np.where(irisY=='setosa')[0],0], real_h[np.where(irisY=='setosa')[0],1], color='red')
plt.scatter(real_h[np.where(irisY=='virginica')[0],0], real_h[np.where(irisY=='virginica')[0],1],color='blue')
plt.scatter(real_h[np.where(irisY=='versicolor')[0],0], real_h[np.where(irisY=='versicolor')[0],1],color='black')
plt.show()

#binary RBM training
for learning_epoch in range(max_epoch):
    rannum=np.random.permutation(len(normalized_irisX))
    num_batch=int(len(normalized_irisX)/mbs)
    for it in range(num_batch):
        batch_X=normalized_irisX[rannum[it*mbs:(it+1)*mbs]]
        w,a,b=Ber_Ber_RBM(batch_X, _w,_a,_b, learning_rate, phase='training')

    if (learning_epoch+1)%100==0:
        print(Ber_Ber_RBM(normalized_irisX, w, a, b, learning_rate, phase='loss'))

binary_h=sigmoid(np.matmul(normalized_irisX, w)+b)

#result scatter
plt.scatter(binary_h[np.where(irisY=='setosa')[0],0], binary_h[np.where(irisY=='setosa')[0],1],color='red')
plt.scatter(binary_h[np.where(irisY=='virginica')[0],0], binary_h[np.where(irisY=='virginica')[0],1],color='blue')
plt.scatter(binary_h[np.where(irisY=='versicolor')[0],0], binary_h[np.where(irisY=='versicolor')[0],1],color='black')
plt.show()

