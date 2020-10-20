import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image as Im
import glob
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data', one_hot=True)

#lr=0.0001 # 0.0001
batch_size=256
z_size=100
n_epoch=10000 #10000 #100000 WGAN_
clip=0.01
alpha=0.2 #
input_dim=784
n_h1=150
n_h2=300
n_critic=5
beta1=0.5#
lr=0.0002#
beta2=0.99#

C_global_step = tf.Variable(0, trainable=False, name='C_global_step') #
G_global_step = tf.Variable(0, trainable=False, name='G_global_step')#

x=tf.placeholder(tf.float32, shape=[batch_size, 28*28], name='X')
z_noise=tf.placeholder(tf.float32, shape=[batch_size, z_size], name='Y')
keep_prob=tf.placeholder(tf.float32, name="keep_prob") #

C_W1=tf.get_variable('C_W1', shape=[input_dim, n_h2], 
initializer=tf.contrib.layers.variance_scaling_initializer())#
#initializer=tf.contrib.layers.xavier_initializer())
C_b1=tf.Variable(tf.zeros([n_h2]), name='C_b1') #random_normal
C_W2=tf.get_variable('C_W2', shape=[n_h2, n_h1],initializer=tf.contrib.layers.variance_scaling_initializer())
C_b2=tf.Variable(tf.zeros([n_h1]), name='C_b2') #random_normal
C_W3=tf.get_variable('C_W3', shape=[n_h1, 1],initializer=tf.contrib.layers.variance_scaling_initializer())
C_b3=tf.Variable(tf.zeros([1]), name='C_b3')#tf.random_normal([1]), name='C_b3')
theta_C=[C_W1, C_b1, C_W2, C_b2, C_W3, C_b3]

G_W1=tf.get_variable('G_W1', shape=[z_size, n_h1],initializer=tf.contrib.layers.variance_scaling_initializer())
G_b1=tf.Variable(tf.zeros([n_h1]), name='G_b1')
G_W2=tf.get_variable('G_W2', shape=[n_h1, n_h2],initializer=tf.contrib.layers.variance_scaling_initializer())
G_b2=tf.Variable(tf.zeros([n_h2]), name='G_b2')
G_W3=tf.get_variable('G_W3', shape=[n_h2, input_dim],initializer=tf.contrib.layers.variance_scaling_initializer())
G_b3=tf.Variable(tf.zeros([input_dim]), name='G_b3')
theta_G=[G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]

def generator(z):
    h1=tf.matmul(z, G_W1)+G_b1
    h1=tf.maximum(alpha*h1, h1)
    h2=tf.matmul(h1, G_W2)+G_b2
    h2=tf.maximum(alpha*h2, h2)
    h3=tf.matmul(h2, G_W3)+G_b3
    out=tf.nn.tanh(h3) ##
    return out

def critic(x):
    h1=tf.matmul(x, C_W1)+C_b1
    h1=tf.maximum(alpha*h1, h1)
    h1=tf.nn.dropout(h1, keep_prob) #
    h2=tf.matmul(h1, C_W2)+C_b2
    h2=tf.maximum(alpha*h2, h2)
    h2=tf.nn.dropout(h2, keep_prob) #
    h3=tf.matmul(h2, C_W3)+C_b3
    #h3=tf.nn.sigmoid(h3)
    return h3

G=generator(z_noise)
C_logit_real=critic(x)
C_logit_fake=critic(G)

C_loss=tf.reduce_mean(C_logit_real)-tf.reduce_mean(C_logit_fake)
G_loss=-tf.reduce_mean(C_logit_fake)

C_solver=tf.train.RMSPropOptimizer(5e-5).minimize(-C_loss, var_list=theta_C)
G_solver=tf.train.RMSPropOptimizer(5e-5).minimize(G_loss, var_list=theta_G)

clip_C=[p.assign(tf.clip_by_value(p, -clip, clip)) for p in theta_C]

config=tf.ConfigProto()
config.gpu_options.allow_growth=True

sess=tf.Session(config=config)
init=tf.global_variables_initializer()

sess.run(init)
z_sample=np.random.uniform(-1,1, size=(batch_size, z_size)).astype(np.float32)

losses = []

for epoch in range(n_epoch):
    for idx in range(5): ##################
        x_value, _=mnist.train.next_batch(batch_size)
        x_value=2*x_value-1
        z_value=np.random.uniform(-1,1,size=(batch_size, z_size))
        C_loss_cur, _, clipC=sess.run([C_loss, C_solver,clip_C], feed_dict={x:x_value, z_noise:z_value,keep_prob:0.7})#keep_prob:0.7
    
    z_value=np.random.uniform(-1,1,size=(batch_size, z_size))
    G_loss_cur, _= sess.run([G_loss, G_solver], feed_dict={x:x_value, z_noise:z_value,keep_prob:0.7}) #
    print("on epoch {}".format(epoch))
    losses.append((C_loss_cur, G_loss_cur))

    if epoch==0 or (epoch+1)%10000==0:
        samples = sess.run(G, feed_dict={z_noise:z_sample})
        samples=0.5*(samples+1)
        for k in range(25):
            print(samples[k])
            plt.subplot(5,5, k+1)
            plt.imshow(np.reshape(samples[k],(28,28)), cmap='gray')
        plt.savefig('./epoch:100000/WGAN_100000/{}.png'.format(str(epoch).zfill(4)),bbox_inches='tight')
        #plt.close()
#loss
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Critic')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses_WGAN")
plt.legend()  

plt.show()