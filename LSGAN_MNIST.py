import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data', one_hot=True)

#parameter settings
n_noise=100
n_h1=150
n_h2=300
batch_size=256
n_epoch=10000
beta1=0.5#
lr=0.0002#
beta2=0.99#

def generator(z_noise):
    #w1=tf.Variable(tf.truncated_normal([n_noise, n_h1], stddev=0.1),name="w1_g", dtype=tf.float32)   
    w1=tf.get_variable('w1_g',shape=[n_noise, n_h1],initializer=tf.contrib.layers.variance_scaling_initializer()) #he
    b1=tf.Variable(tf.zeros([n_h1]), name="b1_g", dtype=tf.float32)
    h1=tf.nn.relu(tf.matmul(z_noise, w1)+b1)

    #w2=tf.Variable(tf.truncated_normal([n_h1, n_h2], stddev=0.1), name="w2_g", dtype=tf.float32)
    w2=tf.get_variable('w2_g',shape=[n_h1, n_h2],initializer=tf.contrib.layers.variance_scaling_initializer())
    b2=tf.Variable(tf.zeros([n_h2]), name="b2_g", dtype=tf.float32)
    h2=tf.nn.relu(tf.matmul(h1, w2)+b2)

    #w3=tf.Variable(tf.truncated_normal([n_h2, 28*28], stddev=0.1), name="w3_g", dtype=tf.float32)    
    w3=tf.get_variable('w3_g',shape=[n_h2, 28*28],initializer=tf.contrib.layers.variance_scaling_initializer())
    b3=tf.Variable(tf.zeros([28*28]), name="b3_g", dtype=tf.float32)
    h3=tf.matmul(h2,w3)+b3

    out_gen=tf.nn.tanh(h3)
    weight_g=[w1,b1,w2,b2,w3,b3]
    return out_gen, weight_g

def discriminator(x, out_gen, keep_prob):
    x_all=tf.concat([x, out_gen], 0)
    
    #w1=tf.Variable(tf.truncated_normal([28*28, n_h2], stddev=0.1), name="w1_d", dtype=tf.float32)
    w1=tf.get_variable('w1_d',shape=[28*28, n_h2],initializer=tf.contrib.layers.variance_scaling_initializer())
    b1=tf.Variable(tf.zeros([n_h2]), name="b1_d", dtype=tf.float32)
    h1=tf.nn.dropout(tf.nn.relu(tf.matmul(x_all, w1)+b1), keep_prob)

    #w2=tf.Variable(tf.truncated_normal([n_h2, n_h1], stddev=0.1), name="w2_d", dtype=tf.float32)
    w2=tf.get_variable('w2_d',shape=[n_h2, n_h1],initializer=tf.contrib.layers.variance_scaling_initializer())
    b2=tf.Variable(tf.zeros([n_h1]), name="b2_d", dtype=tf.float32)
    h2=tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2)+b2), keep_prob)

    #w3=tf.Variable(tf.truncated_normal([n_h1, 1], stddev=0.1), name="w3_d", dtype=tf.float32)
    w3=tf.get_variable('w3_d',shape=[n_h1, 1],initializer=tf.contrib.layers.variance_scaling_initializer())
    b3=tf.Variable(tf.zeros([1]), name="b3_d", dtype=tf.float32)
    h3=tf.matmul(h2, w3)+b3

    y_data=tf.nn.sigmoid(tf.slice(h3, [0,0], [batch_size, -1], name=None))
    y_fake=tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1,1], name=None))
    weight_d=[w1,b1,w2,b2,w3,b3]
    return y_data, y_fake, weight_d

x=tf.placeholder(tf.float32, [batch_size, 28*28], name="x_data")
z_noise=tf.placeholder(tf.float32, [batch_size, n_noise], name="z_prior")

#keep_probability in dropout
keep_prob=tf.placeholder(tf.float32, name="keep_prob")

out_gen, weight_g=generator(z_noise)
y_data, y_fake, weight_d=discriminator(x, out_gen, keep_prob)

#loss function and optimizer for discriminator/generator
d_loss=0.5*(tf.reduce_mean(y_data-1)**2)+0.5*tf.reduce_mean(y_fake**2)
g_loss=0.5*tf.reduce_mean((y_fake-1)**2)

optimizer=tf.train.AdamOptimizer(lr, beta1, beta2) #, beta1, beta2
d_trainer=optimizer.minimize(d_loss, var_list=weight_d) #weight_d update
g_trainer=optimizer.minimize(g_loss, var_list=weight_g) #weight_g update

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
z_sample=np.random.uniform(-1,1, size=(batch_size, n_noise)).astype(np.float32)

losses = []

for epoch in range(n_epoch):
    batch_x, _=mnist.train.next_batch(batch_size)
    x_value=2*batch_x.astype(np.float32)-1
    z_value=np.random.uniform(-1,1,size=(batch_size, n_noise)).astype(np.float32)
    sess.run(d_trainer, feed_dict={x:x_value, z_noise:z_value, keep_prob:0.7})
    sess.run(g_trainer, feed_dict={x:x_value, z_noise:z_value, keep_prob:0.7})
    [c1,c2]=sess.run([d_loss, g_loss], feed_dict={x:x_value, z_noise:z_value, keep_prob:0.7})
    print("on epoch {}".format(epoch))
    losses.append((c1, c2))


    if epoch==0 or (epoch+1)%1000==0:
        samples = sess.run(out_gen, feed_dict={z_noise:z_sample})
        imgs=0.5*(samples+1) 
        for k in range(25):
            plt.subplot(5,5, k+1)
            plt.imshow(np.reshape(imgs[k],(28,28)), cmap='gray')
        plt.savefig('./TEST/{}.png'.format(str(epoch).zfill(4)),bbox_inches='tight')
        #plt.close()

#loss
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses_LSGAN")
plt.legend()  

plt.show()

#https://pathmind.com/kr/wiki/generative-adversarial-network-gan
#http://solarisailab.com/archives/2482
