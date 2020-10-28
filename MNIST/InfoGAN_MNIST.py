import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
# from keras.datasets.mnist import load_data
from tensorflow.examples.tutorials.mnist import input_data

# (train_x,_), (_, _)=load_data()
# sample_size=train_x.shape[0]
# train_x=train_x.reshape(sample_size, 784)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sample_size = 60000

lr = 0.0002
beta1 = 0.5
beta2 = 0.99

batch_size = 20
z_size = 100
C_size = 4
epoch = 100000
g_hidden_size = 150
d_hidden_size = 300
alpha = 0.2

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
Z = tf.placeholder(tf.float32, shape=[None, z_size], name='Z')
C = tf.placeholder(tf.float32, shape=[None, C_size], name='C')

# distriminator weight and bias
D_W1 = tf.get_variable('D_W1', shape=[784, d_hidden_size], initializer=tf.contrib.layers.xavier_initializer())
D_b1 = tf.Variable(tf.random_normal([d_hidden_size]), name='D_b1')
D_W2 = tf.get_variable('D_W2', shape=[d_hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
D_b2 = tf.Variable(tf.random_normal([1]), name='D_b2')
theta_D = [D_W1, D_W2, D_b1, D_b2]

# generator weight and bias
G_W1 = tf.get_variable('G_W1', shape=[z_size + C_size, g_hidden_size],
                       initializer=tf.contrib.layers.xavier_initializer())
G_b1 = tf.Variable(tf.zeros(shape=[g_hidden_size]), name='G_b1')
G_W2 = tf.get_variable('G_W2', shape=[g_hidden_size, 784], initializer=tf.contrib.layers.xavier_initializer())
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')
theta_G = [G_W1, G_W2, G_b1, G_b2]

# Q layer weight and bias
Q_W1 = tf.get_variable('Q_W1', shape=[784, g_hidden_size], initializer=tf.contrib.layers.xavier_initializer())
Q_b1 = tf.Variable(tf.random_normal([g_hidden_size]), name='Q_b1')
Q_W2 = tf.get_variable('Q_W2', shape=[g_hidden_size, C_size], initializer=tf.contrib.layers.xavier_initializer())
Q_b2 = tf.Variable(tf.random_normal([C_size]), name='Q_b2')
theta_Q = [Q_W1, Q_W2, Q_b1, Q_b2]


def generator(z, c):
    with tf.variable_scope('generator'):
        zc = tf.concat([z, c], axis=1)
        h1 = tf.matmul(zc, G_W1) + G_b1
        #h1 = tf.maximum(alpha * h1, h1)
        h1=tf.nn.relu(h1)

        h2 = tf.matmul(h1, G_W2) + G_b2
        return tf.nn.tanh(h2)


def discriminator(x):
    with tf.variable_scope('discriminator'):
        h1 = tf.matmul(x, D_W1) + D_b1
        #h1 = tf.maximum(alpha * h1, h1)
        h1=tf.nn.relu(h1)

        h1 = tf.nn.dropout(h1, keep_prob)
        h2 = tf.matmul(h1, D_W2) + D_b2
        h2 = tf.nn.dropout(h2, keep_prob)
        h3 = tf.nn.sigmoid(h2)
        return h3, h2


def Q(x):
    h1 = tf.matmul(x, Q_W1) + Q_b1
    #h1 = tf.maximum(h1 * alpha, h1)
    h1=tf.nn.relu(h1)

    h2 = tf.nn.dropout(h1, keep_prob)
    h2 = tf.matmul(h2, Q_W2) + Q_b2
    h2 = tf.nn.dropout(h2, keep_prob)
    h3 = tf.nn.softmax(tf.matmul(h1, Q_W2) + Q_b2)
    return h3


G = generator(Z, C)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G)
Q_res = Q(G)

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_fake + D_loss_real

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

MI = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_res + 1e-10) * C, 1))
Q_loss = MI

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if 'D_' in var.name]
G_vars = [var for var in T_vars if 'G_' in var.name]
Q_vars = [var for var in T_vars if 'Q_' in var.name]

D_solver = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(D_loss, var_list=D_vars)
G_solver = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(G_loss, var_list=G_vars)
Q_solver = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(Q_loss, var_list=Q_vars + G_vars)  #, beta1, beta2

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

losses = []
for epoch_ in range(epoch):
    batch_img, _ = mnist.train.next_batch(batch_size)
    # print(batch_x.shape) #256, 784 / 256, 10
    batch_img = 2 * batch_img.astype(np.float32)-1
    batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size)).astype(np.float32)
    c_noise = np.random.multinomial(1, C_size * [1. / C_size], size=batch_size)
    sess.run(D_solver, feed_dict={X: batch_img, Z: batch_z, C: c_noise, keep_prob: 0.7})
    sess.run(G_solver, feed_dict={Z: batch_z, C: c_noise, keep_prob: 0.7})
    sess.run(Q_solver, feed_dict={Z: batch_z, C: c_noise, keep_prob: 0.7})

    D_loss_cur = sess.run(D_loss, feed_dict={X: batch_img, Z: batch_z, C: c_noise, keep_prob: 0.7})
    G_loss_cur = sess.run(G_loss, feed_dict={Z: batch_z, C: c_noise, keep_prob: 0.7})
    Q_loss_cur = sess.run(Q_loss, feed_dict={Z: batch_z, C: c_noise, keep_prob: 0.7})

    print("Epoch:%d" % (epoch_), 'D_loss=%f, G_loss=%f, Q_loss=%f' % (D_loss_cur, G_loss_cur, Q_loss_cur))
    losses.append((D_loss_cur, G_loss_cur, Q_loss_cur))

    if (epoch_ + 1) % 100000 == 0:
        sample_z = np.random.uniform(-1, 1, size=(100, z_size)).astype(np.float32)
        sample_c = np.random.multinomial(1, C_size * [1. / C_size], size=100)
        generated_img = sess.run(G, feed_dict={Z: sample_z, C: sample_c})
        generated_img = 0.5 * (generated_img + 1)
        for k in range(25):
            plt.subplot(5, 5, k + 1)
            plt.imshow(np.reshape(generated_img[k], (28, 28)), cmap='gray')
        plt.savefig('./img/{}.png'.format(str(epoch_).zfill(4)), bbox_inches='tight')

