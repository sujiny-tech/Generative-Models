import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report, auc, roc_curve

df=pd.read_csv('OnlineNewsPopularity.csv', header=0)
print(df)

#DBN for classification(hidden layer 2)

class_=pd.DataFrame(np.where(df.iloc[:,60]>=1400, 1, 0))

df=pd.concat([df, class_], axis=1)

scaler=StandardScaler().fit(df.iloc[:,1:60])
data_s=scaler.transform(df.iloc[:,1:60])

x_train, x_test, y_train, y_test=train_test_split(data_s, df.iloc[:,61], test_size=0.33, random_state=20200101, stratify=df.iloc[:,61])
#print(df.iloc[:,60])

y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)

n_input=59
n_hidden1=100
n_hidden2=40
n_class=2
n_iter=200

lr=0.01
n_epoch=200
batch_size=128

display_step=100

x=tf.placeholder(tf.float32, [None, n_input], name="x")
y=tf.placeholder(tf.float32, [None, n_class], name="y")

#define weight for hidden layer
W1=tf.Variable(tf.random_normal([n_input, n_hidden1], 0.01), name="W1")
b1_h=tf.Variable(tf.zeros([1, n_hidden1], tf.float32, name="b1_h"))
b1_i=tf.Variable(tf.zeros([1, n_input], tf.float32, name="b1_i"))


W2=tf.Variable(tf.random_normal([n_hidden1, n_hidden2], 0.01), name="W2")
b2_h=tf.Variable(tf.zeros([1, n_hidden2], tf.float32, name="b2_h"))
b2_i=tf.Variable(tf.zeros([1, n_hidden1], tf.float32, name="b2_i"))


W_c=tf.Variable(tf.random_normal([n_hidden2, n_class], 0.01), name="W_c")
b_c=tf.Variable(tf.zeros([1, n_class], tf.float32, name="b_c"))

def binary(probs):
    return tf.floor(probs+tf.random_uniform(tf.shape(probs), 0, 1))

def cd_step(x_k, W, b_h, b_i):
    h_k=binary(tf.sigmoid(tf.matmul(x_k, W)+b_h))
    x_k=binary(tf.sigmoid(tf.matmul(h_k, tf.transpose(W))+b_i))
    return x_k

def cd_gibbs(k, x_k,W, b_h, b_i):
    for i in range(k):
        x_out=cd_step(x_k, W, b_h, b_i)
    return x_out


# for new x_s
x_s=cd_gibbs(2, x, W1, b1_h, b1_i)
act_h1_s=binary(tf.sigmoid(tf.matmul(x_s, W1)+b1_h))
h1_s=cd_gibbs(2, act_h1_s, W2, b2_h, b2_i)
act_h2_s=binary(tf.sigmoid(tf.matmul(h1_s, W2)+b2_h))

# for x
act_h1=tf.sigmoid(tf.matmul(x,W1)+b1_h)
act_h2=tf.sigmoid(tf.matmul(act_h1_s, W2)+b2_h)

#weigth, bias update
size_batch=tf.cast(tf.shape(x)[0], tf.float32)
W1_add=tf.multiply(lr/size_batch, tf.subtract(tf.matmul(tf.transpose(x), act_h1), tf.matmul(tf.transpose(x_s), act_h1_s)))
b1_i_add=tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(x, x_s), 0, True))
b1_h_add=tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(act_h1, act_h1_s), 0, True))

W2_add=tf.multiply(lr/size_batch, tf.subtract(tf.matmul(tf.transpose(act_h1_s), act_h2), tf.matmul(tf.transpose(h1_s), act_h2_s)))
b2_i_add=tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(act_h1_s, h1_s), 0, True))
b2_h_add=tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(act_h2, act_h2_s), 0, True))

updt=[W1.assign_add(W1_add), b1_i.assign_add(b1_i_add), b1_h.assign_add(b1_h_add),
      W2.assign_add(W2_add), b2_i.assign_add(b2_i_add), b2_h.assign_add(b2_h_add)]

#operation step for classification-DBN added softmax layer
logits=tf.matmul(act_h2, W_c)+b_c
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost) #
correct_pred=tf.equal(tf.argmax(logits,1), tf.argmax(y, 1))
accuracy=tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    
    n_batch=int(len(x_train)/batch_size)
    #Start the training
    for epoch in range(n_epoch):
        for i in range(n_batch):
            batch_xs=x_train[i*batch_size:(i+1)*batch_size]
            batch_ys=y_train[i*batch_size:(i+1)*batch_size]
            batch_xs=(batch_xs>0)*1
            _=sess.run([updt], feed_dict={x:batch_xs})
        #display running step
        if epoch%display_step==0:
            print("Epoch:", '%04d'%(epoch+1))

    print("RBM training Completed!")

    #train/pred for classification DBN added softmax layer
    for i in range(n_iter):
        batch_x=x_train[i*batch_size:(i+1)*batch_size]
        batch_y=y_train[i*batch_size:(i+1)*batch_size]
        sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})

        if i%10==0:
            tr_loss, tr_acc=sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y})
            print("Iter "+str(i)+", Minibatch_Loss= "+"{:.6f}".format(tr_loss)+", Training_Accuracy= "+"{:.5f}".format(tr_acc))

    print("Optimization Finished!")

    print("Testing_Accuracy:", sess.run(accuracy, feed_dict={x:x_test, y:y_test}))
    sess.close()
    
    
