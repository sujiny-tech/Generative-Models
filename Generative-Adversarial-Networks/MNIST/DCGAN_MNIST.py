import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2#
 
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data', one_hot=True)

total_epoch = 10000
batch_size = 256
n_noise = 100

D_global_step = tf.Variable(0, trainable=False, name='D_global_step') #
G_global_step = tf.Variable(0, trainable=False, name='G_global_step')#

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Z = tf.placeholder(tf.float32, [None, n_noise])
is_training = tf.placeholder(tf.bool)
 
def leaky_relu(x, leak=0.2):
    return tf.maximum(x, x * leak)
 
def generator(noise, reuse=None):
    with tf.variable_scope('generator', reuse=reuse):
        output = tf.layers.dense(noise, 128*7*7)
        output = tf.reshape(output, [-1, 7, 7, 128])
        output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        output = tf.layers.conv2d_transpose(output, 64, [5, 5], strides=(2, 2), padding='SAME')
        output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        output = tf.layers.conv2d_transpose(output, 32, [5, 5], strides=(2, 2), padding='SAME')
        output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        output = tf.layers.conv2d_transpose(output, 1, [5, 5], strides=(1, 1), padding='SAME')
        output = tf.tanh(output)
    return output
 
def discriminator(inputs, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        output = tf.layers.conv2d(inputs, 32, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)
        output = tf.layers.conv2d(output, 64, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(tf.layers.batch_normalization(output, training=is_training))
        output1 = tf.layers.conv2d(output, 128, [5, 5], strides=(2, 2), padding='SAME')
        output2 = leaky_relu(tf.layers.batch_normalization(output1, training=is_training))
        flat = tf.contrib.layers.flatten(output2)
        output = tf.layers.dense(flat, 1, activation=None)
    return output, output1
 
G = generator(Z)
#real data discrimination
D_real, D_logit_real=discriminator(X)
#fake data discrimination
D_fake, D_logit_fake=discriminator(G, reuse=True) #

#loss function
D_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss=D_loss_real+D_loss_fake
G_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

T_vars=tf.trainable_variables()
D_vars=[var for var in T_vars if var.name.startswith('discriminator')]
G_vars=[var for var in T_vars if var.name.startswith('generator')]

#수정
update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    D_solver=tf.train.AdamOptimizer(0.0002, 0.5, 0.99).minimize(D_loss, var_list=D_vars, global_step=D_global_step)
    G_solver=tf.train.AdamOptimizer(0.0002, 0.5, 0.99).minimize(G_loss, var_list=G_vars, global_step=G_global_step)

#D_solver=tf.train.AdamOptimizer(0.0002, 0.5, 0.99).minimize(D_loss, var_list=D_vars) #0.0002, 0.5, 0.99
#G_solver=tf.train.AdamOptimizer(0.0002, 0.5, 0.99).minimize(G_loss, var_list=G_vars)
 #(lr,beta1,beta2)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
losses = []

total_batch = int(mnist.train.num_examples / batch_size)
for epoch in range(total_epoch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape(-1, 28, 28, 1)
    batch_xs=2*batch_xs.astype(np.float32)-1 #
    noise = np.random.uniform(-1,1, size=[batch_size, n_noise])

    D_loss_curr, _=sess.run([D_loss, D_solver], feed_dict={X:batch_xs, Z:noise, is_training: True})
    G_loss_curr, _=sess.run([G_loss, G_solver], feed_dict={X:batch_xs, Z:noise, is_training: True})
    losss=D_loss_curr+G_loss_curr

    print('Epoch:', '%04d' % epoch,'D loss: {:.4}'.format(D_loss_curr),'G loss: {:.4}'.format(G_loss_curr))
    losses.append((D_loss_curr, G_loss_curr))    


    if epoch==0 or (epoch+1)%1000==0:
        sample_size = 25 #
        noise = np.random.uniform(-1.0, 1.0, size=[sample_size, n_noise]) #
        samples = sess.run(G, feed_dict={Z: noise, is_training: False})
        for k in range(25):
            plt.subplot(5,5, k+1)
            plt.imshow(np.reshape(samples[k],(28,28)), cmap='gray')
        plt.savefig('./DCGAN_50000/{}.png'.format(str(epoch).zfill(4)),bbox_inches='tight')

#loss
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses_DCGAN")
plt.legend()  

plt.show()
