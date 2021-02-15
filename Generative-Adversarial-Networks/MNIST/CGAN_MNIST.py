import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data', one_hot=True)

total_epoch=100000
batch_size=256
n_noise=100

n_h1=150
n_h2=300

alpha=0.2
lr=0.0002
beta1=0.5
beta2=0.99

keep_prob=tf.placeholder(tf.float32, name='keep_prob')
Y=tf.placeholder(tf.float32, shape=[None, 10], name='Y')
X=tf.placeholder(tf.float32, shape=[None, 784], name='X')
Z=tf.placeholder(tf.float32, shape=[None, n_noise], name='Z')

G_w1=tf.get_variable('w1_g',shape=[n_noise+10, n_h1],initializer=tf.contrib.layers.variance_scaling_initializer()) #he
G_b1=tf.Variable(tf.zeros([n_h1]), name="b1_g") 
G_w2=tf.get_variable('w2_g',shape=[n_h1, n_h2],initializer=tf.contrib.layers.variance_scaling_initializer()) #he
G_b2=tf.Variable(tf.zeros([n_h2]), name="b2_g")
G_w3=tf.get_variable('w3_g',shape=[n_h2, 784],initializer=tf.contrib.layers.variance_scaling_initializer()) #he
G_b3=tf.Variable(tf.zeros([784]), name="b3_g")
theta_g=[G_w1, G_w2, G_w3, G_b1, G_b2, G_b3]

D_w1=tf.get_variable('w1_d',shape=[784+10, n_h2],initializer=tf.contrib.layers.variance_scaling_initializer()) #he
D_b1=tf.Variable(tf.zeros([n_h2]), name="b1_d")
D_w2=tf.get_variable('w2_d',shape=[n_h2, n_h1],initializer=tf.contrib.layers.variance_scaling_initializer()) #he
D_b2=tf.Variable(tf.zeros([n_h1]), name="b2_d")
D_w3=tf.get_variable('w3_d',shape=[n_h1, 1],initializer=tf.contrib.layers.variance_scaling_initializer()) #he
D_b3=tf.Variable(tf.zeros([1]), name="b3_d")
theta_d=[D_w1, D_w2, D_w3, D_b1, D_b2, D_b3]

def leaky_relu(x, leak=0.2):
    return tf.maximum(x, x * leak)

def generator(z, y):
    zy=tf.concat([z,y], axis=1)
    h1=tf.matmul(zy, G_w1)+G_b1
    h1=tf.nn.relu(h1)
    #h1=leaky_relu(h1)
    
    h2=tf.matmul(h1, G_w2)+G_b2
    h2=tf.nn.relu(h2)
    #h2=leaky_relu(h2)
    
    h3=tf.matmul(h2, G_w3)+G_b3
    output=tf.nn.tanh(h3)
    
    return output

def discriminator(x, y):
    xy=tf.concat([x,y], axis=1)
    h1=tf.matmul(xy, D_w1)+D_b1
    h1=tf.nn.relu(h1)
    #h1=leaky_relu(h1) 
    h1=tf.nn.dropout(h1, keep_prob)

    h2=tf.matmul(h1, D_w2)+D_b2
    h2=tf.nn.relu(h2)
    #h2=leaky_relu(h2) 
    h2=tf.nn.dropout(h2, keep_prob)
    
    h3=tf.matmul(h2, D_w3)+D_b3
    #prob=tf.nn.sigmoid(h3)

    return h3, h2

G=generator(Z,Y)
D_real, D_logit_real=discriminator(X,Y)
D_fake, D_logit_fake=discriminator(G,Y)

D_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss=D_loss_real+D_loss_fake

G_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver=tf.train.AdamOptimizer(lr, beta1,beta2).minimize(D_loss, var_list=theta_d)
G_solver=tf.train.AdamOptimizer(lr, beta1,beta2).minimize(G_loss, var_list=theta_g)

sess=tf.Session();
sess.run(tf.global_variables_initializer())
losses=[]

for epoch in range(total_epoch): 
    #n_batch=int(len(mnist.train.example)/batch_size)
    batch_x, batch_y=mnist.train.next_batch(batch_size)
    #print(batch_x.shape) #256, 784 / 256, 10
    x_value=2*batch_x-1
    z_value=np.random.uniform(-1,1,size=(batch_size, n_noise))

    D_loss_curr, _ = sess.run([D_loss, D_solver], feed_dict={X:x_value, Z:z_value, Y:batch_y, keep_prob:0.7})
    G_loss_curr, _ = sess.run([G_loss, G_solver], feed_dict={X:x_value, Z:z_value, Y:batch_y, keep_prob:0.7})
    losss=D_loss_curr + G_loss_curr
    print("on epoch {}".format(epoch))
    losses.append((D_loss_curr, G_loss_curr))

    if epoch==0 or (epoch+1)%10000==0:
        samples2 = sess.run(G, feed_dict={Z:z_value, Y:batch_y})
        samples2=0.5*(samples2+1)
        for k in range(25):
            plt.subplot(5,5, k+1)
            plt.imshow(np.reshape(samples2[k],(28,28)), cmap='gray')
        plt.savefig('./epoch:100000/CGAN_100000/{}.png'.format(str(epoch).zfill(4)),bbox_inches='tight')

        sample_z=np.random.uniform(-1,1, size=(25, n_noise))
        sample_y=np.zeros(shape=[25, 10])
        sample_y[:,4]=1
        samples = sess.run(G, feed_dict={Z:sample_z,Y:sample_y})
        samples=0.5*(samples+1)
        for k in range(25):
            plt.subplot(5,5, k+1)
            plt.imshow(np.reshape(samples[k],(28,28)), cmap='gray')
        plt.savefig('./epoch:100000/CGAN_100000/4/{}.png'.format(str(epoch).zfill(4)),bbox_inches='tight')
        #plt.close()

#loss
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses_CGAN")
plt.legend()  
plt.savefig('./epoch:100000/CGAN_100000/loss_function.png',bbox_inches='tight')
plt.close()
#plt.show()
