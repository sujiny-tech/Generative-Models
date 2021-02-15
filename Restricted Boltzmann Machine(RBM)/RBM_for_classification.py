import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report, auc, roc_curve

df=pd.read_csv('OnlineNewsPopularity.csv', header=0)
print(df)

#RBM
#binary input for classification
'''
med=df.median()
#print(med) #median 1400

ind1=(df.iloc[:,60]>=1400)
data1=df[ind1]

ind2=(df.iloc[:,60]<1400)
data2=df[ind2]

#print(data1.shape)
#print(data2.shape)
'''
class_=pd.DataFrame(np.where(df.iloc[:,60]>=1400, 1, 0))

df=pd.concat([df, class_], axis=1)

scaler=StandardScaler().fit(df.iloc[:,1:60])
data_s=scaler.transform(df.iloc[:,1:60])

x_train, x_test, y_train, y_test=train_test_split(data_s, df.iloc[:,61], test_size=0.33, random_state=20200101, stratify=df.iloc[:,61])
#print(df.iloc[:,60])

y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)

n_input=59
n_hidden=100
n_class=2

lr=0.01
n_epoch=200
batch_size=128

display_step=10

x=tf.placeholder(tf.float32, [None, n_input], name="x")
y=tf.placeholder(tf.float32, [None, n_class], name="y")

#define weight for hidden layer
W_xh=tf.Variable(tf.random_normal([n_input, n_hidden], 0.01), name="W_xh")
W_hy=tf.Variable(tf.random_normal([n_hidden, n_class], 0.01), name="W_hy")
b_i=tf.Variable(tf.zeros([1, n_input], tf.float32, name="b_i"))
b_h=tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="b_h"))
b_y=tf.Variable(tf.zeros([1, n_class], tf.float32, name="b_y"))

def binary(probs):
    return tf.floor(probs+tf.random_uniform(tf.shape(probs), 0, 1))

#gibbs sampling
def gibbs_step(x_k, y_k):
    h_k=binary(tf.sigmoid(tf.matmul(x_k,W_xh)+tf.matmul(y_k, tf.transpose(W_hy))+b_h))
    x_k=binary(tf.sigmoid(tf.matmul(h_k, tf.transpose(W_xh))+b_i))
    y_k=tf.nn.softmax(tf.matmul(h_k,W_hy)+b_y)
    return x_k, y_k

def gibbs_sample(k, x_k, y_k):
    for i in range(k):
        x_out, y_out=gibbs_step(x_k,y_k)
    return x_out, y_out

#CD-2 algorithm
x_s, y_s=gibbs_sample(2,x,y)

act_h_s=tf.sigmoid(tf.matmul(x_s, W_xh)+tf.matmul(y_s, tf.transpose(W_hy))+b_h)

act_h=tf.sigmoid(tf.matmul(x, W_xh)+tf.matmul(y, tf.transpose(W_hy))+b_h)

_x=(tf.sigmoid(tf.matmul(act_h, tf.transpose(W_xh))+b_i))

W_xh_add=tf.multiply(lr/batch_size, tf.subtract(tf.matmul(tf.transpose(x), act_h), tf.matmul(tf.transpose(x_s), act_h_s)))
W_hy_add=tf.multiply(lr/batch_size, tf.subtract(tf.matmul(tf.transpose(act_h), y), tf.matmul(tf.transpose(act_h_s), y_s)))
bi_add=tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(x,x_s), 0, True))
bh_add=tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(act_h, act_h_s), 0, True))
by_add=tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(y, y_s), 0, True))
updt=[W_xh.assign_add(W_xh_add), W_hy.assign_add(W_hy_add), b_i.assign_add(bi_add), b_h.assign_add(bh_add), b_y.assign_add(by_add)]

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(n_epoch):
        num_batch=int(len(x_train)/batch_size)

        for i in range(num_batch):
            batch_xs=x_train[i*batch_size:(i+1)*batch_size]
            batch_ys=y_train[i*batch_size:(i+1)*batch_size]
            #weight update run
            _=sess.run([updt], feed_dict={x:batch_xs, y:batch_ys})

        if epoch % display_step==0:
            print("Epoch:", '%04d'%(epoch+10))

    print("Discriminatice RBM training Completed !")

    #calculate classification RBM accuracy for train data
    tr_lab1=np.zeros((len(x_train), n_class)); tr_lab1[:,0]=1
    tr_lab2=np.zeros((len(x_train), n_class)); tr_lab2[:,1]=1

    tr_f1_xl=tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(x_train, tf.float32), W_xh)+tf.matmul(tf.cast(tr_lab1, tf.float32),tf.transpose(W_hy))+b_h),1)
    tr_f2_xl=tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(x_train, tf.float32), W_xh)+tf.matmul(tf.cast(tr_lab2, tf.float32),tf.transpose(W_hy))+b_h),1)
    tr_f_xl=b_y+tf.transpose([tr_f1_xl, tr_f2_xl])
    tr_y_hat=tf.nn.softmax(tr_f_xl)

    tr_correct_pred=tf.equal(tf.argmax(tr_y_hat,1), tf.argmax(y_train,1))
    tr_accuracy=tf.reduce_mean(tf.cast(tr_correct_pred, tf.float32))

    print("Training Accuracy:", sess.run(tr_accuracy))

    #calculate classification RBM accuracy for test data
    te_lab1=np.zeros((len(x_test), n_class)); te_lab1[:,0]=1
    te_lab2=np.zeros((len(x_test), n_class)); te_lab2[:,1]=1

    te_f1_xl=tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(x_test, tf.float32), W_xh)+tf.matmul(tf.cast(te_lab1,tf.float32), tf.transpose(W_hy))+b_h),1)
    te_f2_xl=tf.reduce_sum(tf.nn.softplus(tf.matmul(tf.cast(x_test, tf.float32), W_xh)+tf.matmul(tf.cast(te_lab2,tf.float32), tf.transpose(W_hy))+b_h),1)
    te_f_xl=b_y+tf.transpose([te_f1_xl, te_f2_xl])
    te_y_hat=tf.nn.softmax(te_f_xl)

    te_correct_pred=tf.equal(tf.argmax(te_y_hat, 1), tf.argmax(y_test,1))
    te_accuracy=tf.reduce_mean(tf.cast(te_correct_pred, tf.float32))

    print("Test Accuracy:", sess.run(te_accuracy))

    sess.close()
    
