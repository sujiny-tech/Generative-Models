import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
from tensorflow.examples.tutorials.mnist import input_data

from scipy.io import loadmat
import cv2

def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

def plot_images(img, labels, nrows, ncols):
    """ Plot nrows x ncols images
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])

mnist=input_data.read_data_sets('MNIST_data', one_hot=True)

X_train, y_train = load_data('train_32x32.mat')
X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
X_train=X_train[:60000,1:29, 1:29, :]
X_train = rgb2gray(X_train).astype(np.float32)
X_train=(X_train-127.5) / 127.5  ##
X_train=X_train.reshape(-1, 28, 28, 1)

X_test, y_test = load_data('test_32x32.mat')
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]
X_test=X_test[:10000,1:29, 1:29, :]
X_test = rgb2gray(X_test).astype(np.float32)
X_test=(X_test-127.5) / 127.5 ##
X_test=X_test.reshape(-1, 28, 28, 1)

sample_size=60000

#What needs to be fixed : Hyperparameters Optimization
lr=0.0002
beta1=0.5
beta2=0.99
batch_size=300
epoch=1000
alpha=0.2

X_A=tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
X_B=tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

def leaky_relu(x, leak=0.2):
    return tf.maximum(x, x * leak)

def G_AB(X, reuse=None):
    with tf.variable_scope('g_ab', reuse=reuse):
        output = tf.contrib.layers.flatten(X)
        output = tf.layers.dense(output, 11*11*256)
        output = tf.maximum(0.0, output)
        output = tf.reshape(output, [tf.shape(output)[0], 11, 11, 256])
 
        output = tf.keras.layers.UpSampling2D([2,2])(output)
        output = tf.reshape(output, [tf.shape(output)[0], 22, 22, 256])
        output = tf.layers.conv2d(output, 128, [5, 5], strides=(1, 1), padding='VALID')
        output = tf.maximum(0.0, output)

        output = tf.keras.layers.UpSampling2D([2,2])(output)
        output = tf.reshape(output, [tf.shape(output)[0], 36, 36, 128])
        output = tf.layers.conv2d(output, 64, [5, 5], strides=(1, 1), padding='VALID')
        output = tf.maximum(0.0, output)

        output = tf.layers.conv2d(output, 1, [5, 5], strides=(1, 1), padding='VALID')
        output = tf.tanh(output)

        return output

def G_BA(X, reuse=None):
    with tf.variable_scope('g_ba', reuse=reuse):
        output = tf.contrib.layers.flatten(X)
        output = tf.layers.dense(output, 11*11*256)
        output = tf.maximum(0.0, output)
        output = tf.reshape(output, [tf.shape(output)[0], 11, 11, 256])
 
        output = tf.keras.layers.UpSampling2D([2,2])(output)
        output = tf.reshape(output, [tf.shape(output)[0], 22, 22, 256])
        output = tf.layers.conv2d(output, 128, [5, 5], strides=(1, 1), padding='VALID')
        output = tf.maximum(0.0, output)

        output = tf.keras.layers.UpSampling2D([2,2])(output)
        output = tf.reshape(output, [tf.shape(output)[0], 36, 36, 128])
        output = tf.layers.conv2d(output, 64, [5, 5], strides=(1, 1), padding='VALID')
        output = tf.maximum(0.0, output)

        output = tf.layers.conv2d(output, 1, [5, 5], strides=(1, 1), padding='VALID')
        output = tf.tanh(output)

        return output

def D_A(X, reuse=None):
    with tf.variable_scope('d_a', reuse=reuse):
        output = tf.layers.conv2d(X, 64, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)
        
        output = tf.layers.conv2d(output, 128, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)

        output = tf.layers.conv2d(output, 256, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)
        
        output = tf.contrib.layers.flatten(output)
        output1 = tf.layers.dense(output, 1, activation=None)
        output = tf.nn.sigmoid(output1)

        return output

def D_B(X, reuse=None):
    with tf.variable_scope('d_b', reuse=reuse):
        output = tf.layers.conv2d(X, 64, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)
        
        output = tf.layers.conv2d(output, 128, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)

        output = tf.layers.conv2d(output, 256, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)
        
        output = tf.contrib.layers.flatten(output)
        output1 = tf.layers.dense(output, 1, activation=None)
        output = tf.nn.sigmoid(output1)

        return output

X_AB=G_AB(X_A)
X_BA=G_BA(X_B)

#recursive
D_A_real=D_A(X_A)
D_A_fake=D_A(X_BA, reuse=True)

D_B_real=D_B(X_B)
D_B_fake=D_B(X_AB, reuse=True)

X_ABA=G_BA(X_AB, reuse=True)
X_BAB=G_AB(X_BA, reuse=True)

DA_loss=0.5*(tf.reduce_mean((D_A_real-1)**2)+0.5*tf.reduce_mean(D_A_fake**2))
DB_loss=0.5*(tf.reduce_mean((D_B_real-1)**2)+0.5*tf.reduce_mean(D_B_fake**2))
GAB_loss=0.5*tf.reduce_mean((D_A_fake-1)**2)+0.5*tf.reduce_mean((D_B_fake-1)**2)
C_loss=tf.reduce_mean(tf.abs(X_A-X_ABA))+tf.reduce_mean(tf.abs(X_B-X_BAB))
G_loss=C_loss+GAB_loss


T_vars=tf.trainable_variables()
DA_vars=[x for x in tf.trainable_variables() if x.name.startswith('d_a')]
DB_vars=[x for x in tf.trainable_variables() if x.name.startswith('d_b')]
G_vars=[x for x in tf.trainable_variables() if x.name.startswith('g_ab')]+[x for x in tf.trainable_variables() if x.name.startswith('g_ba')]


DA_solver=tf.train.AdamOptimizer(lr,beta1, beta2).minimize(DA_loss, var_list=DA_vars)
DB_solver=tf.train.AdamOptimizer(lr,beta1, beta2).minimize(DB_loss, var_list=DB_vars)
G_solver=tf.train.AdamOptimizer(lr,beta1, beta2).minimize(G_loss, var_list=G_vars)

init=tf.global_variables_initializer()

#gpu
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
sess.run(init)

losses=[]
for epoch_ in range(epoch):
    batch_A, _=mnist.train.next_batch(batch_size)
    batch_A=batch_A.reshape(-1, 28, 28, 1)

    batch_B= X_train[epoch_*batch_size:(epoch_+1)*batch_size] #

    batch_A = 2 * batch_A.astype(np.float32)-1.0 #

    DA_loss_cur,_=sess.run([DA_loss, DA_solver], feed_dict={X_A:batch_A, X_B:batch_B})
    DB_loss_cur,_=sess.run([DB_loss, DB_solver], feed_dict={X_A:batch_A, X_B:batch_B})
    G_loss_cur,_=sess.run([G_loss, G_solver], feed_dict={X_A:batch_A, X_B:batch_B})
    losses.append((DA_loss_cur, DB_loss_cur, G_loss_cur))
    print("Epoch:%d" % (epoch_), 'DA_loss=%f, DB_loss=%f, G_loss=%f' % (DA_loss_cur, DB_loss_cur, G_loss_cur))

    if (epoch_+1)%100==0:
        test_mnist, _=mnist.test.next_batch(100)
        test_mnist=test_mnist.reshape(-1, 28, 28, 1)
        sample_A=sess.run(X_BA,feed_dict={X_B:X_test[:100]})
        sample_B=sess.run(X_AB,feed_dict={X_A:test_mnist})

        sample_A = 0.5 * (sample_A + 1) #
        sample_B = 0.5 * (sample_B + 1) #

        for k in range(25):
            plt.subplot(5, 5, k + 1)
            plt.imshow(np.reshape(sample_A[k], (28, 28)), cmap='gray')
        plt.savefig('./img/A_{}.png'.format(str(epoch_).zfill(4)),bbox_inches='tight')

        for k in range(25):
            plt.subplot(5, 5, k + 1)
            plt.imshow(np.reshape(sample_B[k], (28, 28)), cmap='gray')
        plt.savefig('./img/B_{}.png'.format(str(epoch_).zfill(4)),bbox_inche='tight')
