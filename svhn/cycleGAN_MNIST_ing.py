#pIx2pix는 cgan의 일종, 이미지 변환에 폭넓게 사용되면서 픽스투픽스라는 이름을 갖게된것
#but, pix2pix 학습시 이미지쌍이 필요한데, 사실 현실에서 쌍으로 구성된 이미지를 항상 얻을수 있는건 아님.
#어떤 도메인에 속하는 데이터를 구하는 것.

#어떤 도메인에 속하는 이미지 x를 입력하여 다른 도메인에 속하는 이미지 y로 변환하는 방법이 개발- cycleGAN
#모드붕괴 방지하려고, 생성한 이미지 G(x)를 입력된 이미지 x로 다시 변환할수있어야함. -> 즉, F(G(x))=x
#즉, 입력이미지를 변환하되 복구가 가능하도록 변환해야한다.
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
X_train=X_train[:,1:29, 1:29, :]
X_train = rgb2gray(X_train).astype(np.float32)
X_train=X_train.reshape(-1, 784)

X_test, y_test = load_data('test_32x32.mat')
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]
X_test=X_test[:,1:29, 1:29, :]
X_test = rgb2gray(X_test).astype(np.float32)
X_test=X_test.reshape(-1, 784)

sample_size=60000

lr=0.001
beta1=0.5
beta2=0.99
batch_size=64
z_size=100
C_size=4
epoch=100000
h_dim=300
h_dim2=150

X_A=tf.placeholder(tf.float32, shape=[None, 784])
X_B=tf.placeholder(tf.float32, shape=[None, 784])

D_A_W1=tf.get_variable('D_A_W1', shape=[784, h_dim], initializer=tf.contrib.layers.xavier_initializer())
D_A_b1=tf.Variable(tf.random_normal([h_dim]))
D_A_W2=tf.get_variable('D_A_W2', shape=[h_dim, h_dim2], initializer=tf.contrib.layers.xavier_initializer())
D_A_b2=tf.Variable(tf.random_normal([h_dim2]))
D_A_W3=tf.get_variable('D_A_W3', shape=[h_dim2, 1], initializer=tf.contrib.layers.xavier_initializer())
D_A_b3=tf.Variable(tf.random_normal([1]))

D_B_W1=tf.get_variable('D_B_W1', shape=[784, h_dim], initializer=tf.contrib.layers.xavier_initializer())
D_B_b1=tf.Variable(tf.random_normal([h_dim]))
D_B_W2=tf.get_variable('D_B_W2', shape=[h_dim, h_dim2], initializer=tf.contrib.layers.xavier_initializer())
D_B_b2=tf.Variable(tf.random_normal([h_dim2]))
D_B_W3=tf.get_variable('D_B_W3', shape=[h_dim2, 1], initializer=tf.contrib.layers.xavier_initializer())
D_B_b3=tf.Variable(tf.random_normal([1]))

G_AB_W1=tf.get_variable('G_AB_W1', shape=[784, h_dim], initializer=tf.contrib.layers.xavier_initializer())
G_AB_b1=tf.Variable(tf.zeros([h_dim]))
G_AB_W2=tf.get_variable('G_AB_W2', shape=[h_dim, h_dim2], initializer=tf.contrib.layers.xavier_initializer())
G_AB_b2=tf.Variable(tf.zeros([h_dim2]))
G_AB_W3=tf.get_variable('G_AB_W3', shape=[h_dim2, 784], initializer=tf.contrib.layers.xavier_initializer())
G_AB_b3=tf.Variable(tf.zeros([784]))

G_BA_W1=tf.get_variable('G_BA_W1', shape=[784, h_dim], initializer=tf.contrib.layers.xavier_initializer())
G_BA_b1=tf.Variable(tf.zeros([h_dim]))
G_BA_W2=tf.get_variable('G_BA_W2', shape=[h_dim, h_dim2], initializer=tf.contrib.layers.xavier_initializer())
G_BA_b2=tf.Variable(tf.zeros([h_dim2]))
G_BA_W3=tf.get_variable('G_BA_W3', shape=[h_dim2, 784], initializer=tf.contrib.layers.xavier_initializer())
G_BA_b3=tf.Variable(tf.zeros([784]))

theta_DA=[D_A_W1, D_A_W2, D_A_W3, D_A_b1, D_A_b2, D_A_b3]
theta_DB=[D_B_W1, D_B_W2, D_B_W3, D_B_b1, D_B_b2, D_B_b3]
theta_G=[G_AB_W1, G_AB_W2, G_AB_W3, G_AB_b1, G_AB_b2, G_AB_b3,
          G_BA_W1, G_BA_W2, G_BA_W3, G_BA_b1, G_BA_b2, G_BA_b3]

def G_AB(X):
    h1=tf.nn.relu(tf.matmul(X, G_AB_W1)+G_AB_b1)
    h1=tf.nn.relu(tf.matmul(h1,G_AB_W2)+G_AB_b2)
    
    return tf.nn.tanh(tf.matmul(h1,G_AB_W3)+G_AB_b3)

def G_BA(X):
    h1=tf.nn.relu(tf.matmul(X, G_BA_W1)+G_BA_b1)
    h1=tf.nn.relu(tf.matmul(h1,G_BA_W2)+G_BA_b2)
    
    return tf.nn.tanh(tf.matmul(h1,G_BA_W3)+G_BA_b3)

def D_A(X):
    h1=tf.nn.relu(tf.matmul(X, D_A_W1)+D_A_b1)
    h1=tf.nn.relu(tf.matmul(h1,D_A_W2)+D_A_b2)
    h1=tf.nn.dropout(h1, 0.7)
    
    return tf.nn.sigmoid(tf.matmul(h1,D_A_W3)+D_A_b3)

def D_B(X):
    h1=tf.nn.relu(tf.matmul(X, D_B_W1)+D_B_b1)
    h1=tf.nn.relu(tf.matmul(h1,D_B_W2)+D_B_b2)
    h1=tf.nn.dropout(h1, 0.7)
    
    return tf.nn.sigmoid(tf.matmul(h1,D_B_W3)+D_B_b3)

X_AB=G_AB(X_A)
X_BA=G_BA(X_B)

#recursive
D_A_real=D_A(X_A)
D_A_fake=D_A(X_BA)

D_B_real=D_B(X_B)
D_B_fake=D_B(X_AB)

X_ABA=G_BA(X_AB)
X_BAB=G_AB(X_BA)

DA_loss=0.5*(tf.reduce_mean((D_A_real-1)**2)+0.5*tf.reduce_mean(D_A_fake**2))
DB_loss=0.5*(tf.reduce_mean((D_B_real-1)**2)+0.5*tf.reduce_mean(D_B_fake**2))
GAB_loss=0.5*tf.reduce_mean((D_A_fake-1)**2)+0.5*tf.reduce_mean((D_B_fake-1)**2)
C_loss=tf.reduce_mean(tf.abs(X_A-X_ABA))+tf.reduce_mean(tf.abs(X_B-X_BAB))
G_loss=C_loss+GAB_loss

DA_solver=tf.train.AdamOptimizer(0.00001).minimize(DA_loss, var_list=theta_DA)
DB_solver=tf.train.AdamOptimizer(0.00001).minimize(DB_loss, var_list=theta_DB)
G_solver=tf.train.AdamOptimizer(lr,beta1, beta2).minimize(G_loss, var_list=theta_G)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

losses=[]
for epoch_ in range(epoch):
    batch_A, _=mnist.train.next_batch(batch_size)
    batch_B= X_train[epoch_*batch_size:(epoch_+1)*batch_size] #

    DA_loss_cur,_=sess.run([DA_loss, DA_solver], feed_dict={X_A:batch_A, X_B:batch_B})
    DB_loss_cur,_=sess.run([DB_loss, DB_solver], feed_dict={X_A:batch_A, X_B:batch_B})
    G_loss_cur,_=sess.run([G_loss, G_solver], feed_dict={X_A:batch_A, X_B:batch_B})
    losses.append((DA_loss_cur, DB_loss_cur, G_loss_cur))
    print("Epoch:%d" % (epoch_), 'DA_loss=%f, DB_loss=%f, G_loss=%f' % (DA_loss_cur, DB_loss_cur, G_loss_cur))

    if (epoch_+1)%100==0:
        test_mnist, _=mnist.test.next_batch(100)
        sample_A=sess.run(X_BA,feed_dict={X_B:X_test[:100]})
        sample_B=sess.run(X_AB,feed_dict={X_A:test_mnist})

        for k in range(25):
            plt.subplot(5, 5, k + 1)
            plt.imshow(np.reshape(sample_A[k], (28, 28)), cmap='gray')
        plt.savefig('./img/A_{}.png'.format(str(epoch_).zfill(4)),bbox_inches='tight')

        for k in range(25):
            plt.subplot(5, 5, k + 1)
            plt.imshow(np.reshape(sample_B[k], (28, 28)), cmap='gray')
        plt.savefig('./img/B_{}.png'.format(str(epoch_).zfill(4)),bbox_inche='tight')
