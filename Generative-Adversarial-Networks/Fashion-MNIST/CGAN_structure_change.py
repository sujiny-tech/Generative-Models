import tensorflow as tf
from keras.datasets.fashion_mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
#from keras.layers import Embedding

def image_save(data, n, name):
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(data[i], cmap='gray_r')
    plt.savefig('./CGAN_after/{}.png'.format(str(name)), bbox_inches='tight')     

def generated_img(data, n):
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(data[i, :, :, 0], cmap='gray_r')
    plt.savefig('./CGAN_after/generated_img_ver2.png', bbox_inches='tight')     


def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

(trainx, trainy), (_,_)=load_data()

print('Train', trainx.shape, trainy.shape) #(60000, 28, 28) (60000,)

##
trainx=np.expand_dims(trainx, axis=-1)
trainx.astype('float32')
trainx = (trainx-127.5) / 127.5

trainx = trainx.reshape((len(trainx), trainx.shape[1], trainx.shape[2], 1))
trainy = np_utils.to_categorical(trainy, 10) #tf.one_hot(trainy, depth=10, dtype=tf.float32)#
#print('Train', trainx.shape, trainy.shape) #(60000, 28, 28) (60000,)
#trainy =trainy.reshape((len(trainy), 1)) 
print(trainy)

##
total_epoch=100
batch_size=128
n_noise=100

alpha=0.2
lr=0.0002
beta1=0.5
beta2=0.99
n_h1=500

##'t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'

keep_prob=tf.placeholder(tf.float32, name='keep_prob')
Y=tf.placeholder(tf.float32, shape=[None, 10], name='Y')
X=tf.placeholder(tf.float32, shape=[None, 28,28,1], name='X')
Z=tf.placeholder(tf.float32, shape=[None, n_noise], name='Z')

###
def leaky_relu(x, leak=0.2):
    return tf.maximum(x, x * leak)

def generator(z, y, reuse=None):
    with tf.variable_scope('generator', reuse=reuse):
        #cgan_fashion_mnist structure
        #y_=tf.layers.dense(y, 7*7*1) ##(none, 1, 49)
        #y_=tf.reshape(y_, [-1,7,7,1]) ##(none, 7, 7, 1)

        #z_=tf.layers.dense(z, 7*7*128) ##(none, 6272)
        #z_=leaky_relu(z_) ##(none, 6272)
        #z_=tf.reshape(z_, [-1,7, 7, 128]) ##(none, 7, 7, 128)
        #zy=tf.concat([z_,y_], axis=3)

        #output=tf.layers.conv2d_transpose(zy, 128, [4,4], strides=(2,2), padding='SAME')
        #output=leaky_relu(output)

        #output=tf.layers.conv2d_transpose(output, 128, [4,4], strides=(2,2), padding='SAME')
        #output=leaky_relu(output)

        #output=tf.layers.conv2d(output, 1, [7,7], activation='tanh', padding='SAME')

        output = tf.contrib.layers.flatten(z)
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

def discriminator(x, y, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        #cgan_fashion_mnist structure
        #y_=tf.layers.dense(y, 28*28*1)
        #y_=tf.reshape(y_, [-1, 28, 28, 1])

        #xy=tf.concat([x,y_], axis=3)

        #output=tf.layers.conv2d(xy, 128, [3,3], strides=(2,2), padding='SAME')
        #output=leaky_relu(output)

        #output=tf.layers.conv2d(output, 128, [3,3], strides=(2,2), padding='SAME')
        #output=leaky_relu(output)

        #output = tf.contrib.layers.flatten(output)
        #output1=tf.nn.dropout(output, 0.4)

        #output=tf.layers.dense(output1, 1, activation=None)

        output = tf.layers.conv2d(x, 64, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)
        print("output shape: ", output.shape)#14*14*64
        
        output = tf.layers.conv2d(output, 128, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)
        print("output shape: ", output.shape)#7*7*128

        output = tf.layers.conv2d(output, 256, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)
        print("output shape: ", output.shape)#4*4*256
        
        output = tf.contrib.layers.flatten(output)
        output1 = tf.layers.dense(output, 1, activation=None)
        output = tf.nn.sigmoid(output1)
        print("critic _ output shape: ", output.shape)

        return output, output1

G=generator(Z,Y)
D_real, D_logit_real=discriminator(X,Y)
D_fake, D_logit_fake=discriminator(G,Y,reuse=True)

D_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss=D_loss_real+D_loss_fake
G_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

####
D_global_step = tf.Variable(0, trainable=False, name='D_global_step') #
G_global_step = tf.Variable(0, trainable=False, name='G_global_step') #

T_vars=tf.trainable_variables()
D_vars=[var for var in T_vars if var.name.startswith('discriminator')]
G_vars=[var for var in T_vars if var.name.startswith('generator')]

update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    D_solver=tf.train.AdamOptimizer(lr,beta1).minimize(D_loss, var_list=D_vars, global_step=D_global_step)
    G_solver=tf.train.AdamOptimizer(lr,beta1).minimize(G_loss, var_list=G_vars, global_step=G_global_step)

config=tf.ConfigProto()
config.gpu_options.allow_growth=True

sess=tf.Session(config=config); #()
sess.run(tf.global_variables_initializer())
losses=[]

saver=tf.train.Saver()

#new_saver=tf.train.import_meta_graph('./CGAN_after/ckpt/generator_model_010.ckpt.meta')
#new_saver.restore(sess, tf.train.latest_checkpoint('./CGAN_after/ckpt/'))

bat_=int(trainx.shape[0]/batch_size) #60000/128=468
half_b=int(batch_size/2) #128/2 =64

for epoch in range(0,total_epoch):  #100
    for batch in range(bat_): #468
        #n_batch=int(len(mnist.train.example)/batch_size) #64
        batch_x, batch_y=next_batch(half_b, trainx, trainy)
        z_value=np.random.uniform(-1,1,size=(half_b, n_noise))
        D_loss_curr, _ = sess.run([D_loss, D_solver], feed_dict={X:batch_x, Z:z_value, Y:batch_y})
        G_loss_curr, _ = sess.run([G_loss, G_solver], feed_dict={X:batch_x, Z:z_value, Y:batch_y})
        losss=D_loss_curr + G_loss_curr
        print('>%d, %d/%d, d=%.3f g=%.3f' %(epoch+1, batch+1, bat_, D_loss_curr, G_loss_curr))
    print('Epoch:', '%04d' % int(epoch+1),'D loss: {:.4}'.format(D_loss_curr),'G loss: {:.4}'.format(G_loss_curr))
    losses.append((D_loss_curr, G_loss_curr))
    
    #saver.save(sess, './CGAN_after/ckpt/generator_model_%03d.ckpt' % (epoch+1))

    if epoch==0 or (epoch+1)%1==0:
        samples2 = sess.run(G, feed_dict={Z:z_value, Y:batch_y})
        samples2=(samples2+1)/2.0 #####
        for k in range(25):
            plt.subplot(5,5, k+1)
            plt.axis('off')
            plt.imshow(samples2[k].reshape((28,28)), cmap='gray_r') #'gray'
        plt.savefig('./CGAN_after/{}.png'.format(str(epoch+1).zfill(4)),bbox_inches='tight')
        plt.close()

        #sample_z=np.random.uniform(-1,1, size=(half_b, n_noise))
        sample_y=np.zeros(shape=[half_b, 10])
        sample_y[:,6]=1
        samples = sess.run(G, feed_dict={Z:z_value,Y:sample_y})
        samples=(samples+1)/2.0 #####
        for k in range(25):
            plt.subplot(5,5, k+1)
            plt.axis('off')
            plt.imshow(samples[k].reshape((28,28)), cmap='gray_r')
        plt.savefig('./CGAN_after/sandal/{}.png'.format(str(epoch+1).zfill(4)),bbox_inches='tight')
        plt.close()


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
    plt.imshow(sam_z[i].reshape((28,28)), cmap='gray_r')

plt.savefig('./CGAN_after/generate_fashion.png',bbox_inches='tight')
plt.close()

#loss
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses_CGAN")
plt.legend()  
plt.savefig('./CGAN_after/loss_function.png',bbox_inches='tight')
plt.close()
