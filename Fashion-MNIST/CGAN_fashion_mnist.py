import tensorflow as tf
from keras.datasets.fashion_mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

def image_save(data, n, name):
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(data[i], cmap='gray_r')
    plt.savefig('./CGAN/{}.png'.format(str(name)), bbox_inches='tight')     

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

(trainx, trainy), (testx, testy)=load_data()

print('Train', trainx.shape, trainy.shape) #(60000, 28, 28) (60000,)
print('Test', testx.shape, testy.shape) #(10000, 28, 28) (10000,)

image_save(trainx, 10, 'original_image')

##
trainx = trainx.astype('float32') / 255.
#testx = testx.astype('float32') / 255.
trainx = trainx.reshape((len(trainx), np.prod(trainx.shape[1:])))
#testx = testx.reshape((len(testx), np.prod(testx.shape[1:])))
trainy = np_utils.to_categorical(trainy, 10) #tf.one_hot(trainy, depth=10, dtype=tf.float32)#
##
total_epoch=1000
batch_size=256
n_noise=100

n_h1=150
n_h2=300

alpha=0.2
lr=0.0002
beta1=0.5
beta2=0.99

##'t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'

keep_prob=tf.placeholder(tf.float32, name='keep_prob')
Y=tf.placeholder(tf.float32, shape=[None, 10], name='Y')
X=tf.placeholder(tf.float32, shape=[None, 784], name='X')
Z=tf.placeholder(tf.float32, shape=[None, n_noise], name='Z')

###
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
    for batch in range(100):
        #n_batch=int(len(mnist.train.example)/batch_size)
        batch_x, batch_y=next_batch(batch_size, trainx, trainy)
        z_value=np.random.uniform(-1,1,size=(batch_size, n_noise))

        D_loss_curr, _ = sess.run([D_loss, D_solver], feed_dict={X:batch_x, Z:z_value, Y:batch_y, keep_prob:0.7})
        G_loss_curr, _ = sess.run([G_loss, G_solver], feed_dict={X:batch_x, Z:z_value, Y:batch_y, keep_prob:0.7})
        losss=D_loss_curr + G_loss_curr
        print('>%d, %d/%d, d=%.3f g=%.3f' %(epoch+1, batch+1, 100, D_loss_curr, G_loss_curr))
    print('Epoch:', '%04d' % epoch,'D loss: {:.4}'.format(D_loss_curr),'G loss: {:.4}'.format(G_loss_curr))
    losses.append((D_loss_curr, G_loss_curr))

    if epoch==0 or (epoch+1)%100==0:
        samples2 = sess.run(G, feed_dict={Z:z_value, Y:batch_y})
        for k in range(25):
            plt.subplot(5,5, k+1)
            plt.imshow(np.reshape(samples2[k],(28,28)), cmap='gray_r') #'gray'
        plt.savefig('./CGAN/{}.png'.format(str(epoch).zfill(4)),bbox_inches='tight')

        sample_z=np.random.uniform(-1,1, size=(25, n_noise))
        sample_y=np.zeros(shape=[25, 10])
        sample_y[:,4]=1
        samples = sess.run(G, feed_dict={Z:sample_z,Y:sample_y})
        for k in range(25):
            plt.subplot(5,5, k+1)
            plt.imshow(np.reshape(samples[k],(28,28)), cmap='gray_r')
        plt.savefig('./CGAN/coat/{}.png'.format(str(epoch).zfill(4)),bbox_inches='tight')
        #plt.close()

sz_=np.random.uniform(-1,1, size=(100, n_noise))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.axis('off')
    sy_=np.zeros(shape=[100, 10])
    j=int((i)/10)
    if j==10:
        j=j-1
        sy_[:,j]=1
    else :
        sy_[:,j]=1
    sam_z = sess.run(G, feed_dict={Z:sz_, Y:sy_})
    plt.imshow(np.reshape(sam_z[i], (28, 28)), cmap='gray_r')

plt.savefig('./CGAN/generate_fashion.png',bbox_inches='tight')
plt.close()

#loss
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses_CGAN")
plt.legend()  
plt.savefig('./CGAN/loss_function.png',bbox_inches='tight')
plt.close()