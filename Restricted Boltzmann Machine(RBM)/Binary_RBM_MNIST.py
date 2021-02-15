import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data", one_hot=True)

#parameter setting
n_input=784
n_hidden=500
display_step=1
num_epochs=100
batch_size=256
lr=tf.constant(0.001, tf.float32)

x=tf.placeholder(tf.float32, [None, n_input], name="x")
W=tf.Variable(tf.random_normal([n_input, n_hidden], 0.01), name="W")
b_h=tf.Variable(tf.zeros([1,n_hidden], tf.float32, name="b_h"))
b_i=tf.Variable(tf.zeros([1,n_input], tf.float32, name="b_i"))

def binary(probs):
    return tf.floor(probs+tf.random_uniform(tf.shape(probs), 0, 1))

#Gibbs sampling step
def cd_step(x_k):
    h_k=binary(tf.sigmoid(tf.matmul(x_k,W)+b_h))
    x_k=binary(tf.sigmoid(tf.matmul(h_k, tf.transpose(W))+b_i))
    return x_k

#CD-k
def cd_gibbs(k, x_k):
    for i in range(k):
        x_out=cd_step(x_k)
    return x_out

#CD-2 algorithm
x_s=cd_gibbs(2,x)
act_h_s=tf.sigmoid(tf.matmul(x_s,W)+b_h)

act_h=tf.sigmoid(tf.matmul(x,W)+b_h)
_x=binary(tf.sigmoid(tf.matmul(act_h, tf.transpose(W))+b_i))

W_add=tf.multiply(lr/batch_size, tf.subtract(tf.matmul(tf.transpose(x), act_h), tf.matmul(tf.transpose(x_s), act_h_s)))
bi_add=tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(x,x_s), 0, True))
bh_add=tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(act_h, act_h_s), 0, True))
updt=[W.assign_add(W_add), b_i.assign_add(bi_add), b_h.assign_add(bh_add)]

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    
    total_batch=int(mnist.train.num_examples/batch_size)
    for epoch in range(num_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys=mnist.train.next_batch(batch_size)
            batch_xs=(batch_xs>0)*1
            _=sess.run([updt], feed_dict={x:batch_xs})
            
        if epoch%display_step==0:
            print("Epoch:", '%04d'%(epoch+1))
    print("RBM training Completed!")

    out=sess.run(act_h, feed_dict={x:(mnist.test.images[:20]>0)*1})
    label=mnist.test.labels[:20]

    #20 real validation images plot
    plt.figure(1)
    for k in range(20):
        plt.subplot(4,5,k+1)
        image=(mnist.test.images[k]>0)*1
        image=np.reshape(image, (28,28))
        plt.imshow(image, cmap='gray')
    #20 generate validation images plot
    plt.figure(2)
    for k in range(20):
        plt.subplot(4,5,k+1)
        image=sess.run(_x, feed_dict={act_h:np.reshape(out[k], (-1, n_hidden))})
        image=np.reshape(image, (28,28))
        plt.imshow(image, cmap='gray')
        print(np.argmax(label[k]))
    W_out=sess.run(W)

    plt.show()
    sess.close()