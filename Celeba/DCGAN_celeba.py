import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import cv2
from os import listdir
from mtcnn.mtcnn import MTCNN

def load_image(filename):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    return pixels

def extract_face(model, pixels, required_size=(28,28)):
    # detect face in the image
    faces = model.detect_faces(pixels)
    # skip cases where we could not detect a face
    if len(faces) == 0:
        return None
    # extract details of the face
    x1, y1, width, height = faces[0]['box']
    # force detected pixel values to be positive (bug fix)
    x1, y1 = abs(x1), abs(y1)
    # convert into coordinates
    x2, y2 = x1 + width, y1 + height
    # retrieve face pixels
    face_pixels = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face_pixels)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def load_faces(directory, n_faces):
    # prepare model
    model = MTCNN()
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # load the image
        pixels = load_image(directory + filename)
        # get face
        face = extract_face(model, pixels)
        if face is None:
            continue
        # store
        faces.append(face)
        print(len(faces), face.shape)
        # stop once we have enough
        if len(faces) >= n_faces:
            break
    return np.asarray(faces)

def next_batch(num, data):
  idx = np.arange(0 , len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[ i] for i in idx]

  return np.asarray(data_shuffle)

def load_celeba(filename):
    data=np.load(filename)
    x=data['arr_0']
    x=x.astype('float32')
    #scale [0,255] -> [-1,1]
    x=(x-127.5)/127.5
    return x

def plot_faces(faces, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(faces[i])
    plt.savefig('./celeba_result/origin2_ver2_no_scale/after 100/plot_generated_{}.png'.format(str(n)),bbox_inches='tight')
    plt.close()

def save_plot(examples, epoch, n=5):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    plt.savefig('./celeba_result/origin2_ver2_no_scale/after 100/{}.png'.format(str(epoch).zfill(4)),bbox_inches='tight')
    plt.close()

def plot_generated(examples, n):
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :])
    plt.savefig('./celeba_result/origin2_ver2_no_scale/after 100/plot_generated_{}.png'.format(str(examples)),bbox_inches='tight')
    plt.title("interpolate_DCGAN")

Ximg=load_celeba('size28_data_50000.npz')
sample_size=Ximg.shape[0]
input_dim=Ximg.shape[1]
print(Ximg.shape)
Ximg=(Ximg+1)/2.0
plot_faces(Ximg, 10)
plot_faces(Ximg, 5)

##
total_epoch = 100 
batch_size = 128 ##
n_noise = 100
lr=0.0002
beta1=0.5

D_global_step = tf.Variable(0, trainable=False, name='D_global_step') #
G_global_step = tf.Variable(0, trainable=False, name='G_global_step') #

X = tf.placeholder(tf.float32, [None, 28, 28, 3])
Z = tf.placeholder(tf.float32, [None, n_noise])
is_training = tf.placeholder(tf.bool)
 
def leaky_relu(x, leak=0.2):
    return tf.maximum(x, x * leak)
 
def generator(noise, reuse=None):
    with tf.variable_scope('generator', reuse=reuse):
        output = tf.layers.dense(noise, 1024*4*4)
        output = tf.reshape(output, [-1, 4, 4, 1024])
        output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))

        output = tf.layers.conv2d_transpose(output, 512, [4, 4], strides=(1,1), padding='valid')
        output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))

        output = tf.layers.conv2d_transpose(output, 256, [5, 5], strides=(2, 2), padding='SAME')
        output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))

        output = tf.layers.conv2d_transpose(output, 3, [5, 5], strides=(2, 2), padding='SAME')
        output = tf.tanh(output)
    return output
 
def discriminator(inputs, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        output = tf.layers.conv2d(inputs, 64, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(output)
        output = tf.layers.conv2d(output, 128, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(tf.layers.batch_normalization(output, training=is_training))
        
        output = tf.layers.conv2d(output, 256, [5, 5], strides=(2, 2), padding='SAME')
        output = leaky_relu(tf.layers.batch_normalization(output, training=is_training))

        output1 = tf.contrib.layers.flatten(output)
        output = tf.layers.dense(output1, 1, activation=None)
    return output, output1

G = generator(Z)
D_real, D_logit_real=discriminator(X)
D_fake, D_logit_fake=discriminator(G, reuse=True) #

#loss function
D_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss=D_loss_real+D_loss_fake
G_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

T_vars=tf.trainable_variables()
D_vars=[var for var in T_vars if var.name.startswith('discriminator')]
G_vars=[var for var in T_vars if var.name.startswith('generator')]

update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    D_solver=tf.train.AdamOptimizer(lr,beta1).minimize(D_loss, var_list=D_vars, global_step=D_global_step)
    G_solver=tf.train.AdamOptimizer(lr,beta1).minimize(G_loss, var_list=G_vars, global_step=G_global_step)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
losses = []

saver=tf.train.Saver()

###load train model
new_saver=tf.train.import_meta_graph('./celeba_result/origin2_ver2_no_scale/generator_model_100.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./celeba_result/origin2_ver2_no_scale/'))

total_batch = int(sample_size / batch_size) #50000/128 =390
half_batch=int(batch_size/2) #195
total_epoch=100

for epoch in range(total_epoch, 200):#100
    for batch in range(total_batch): #390
        batch_xs=next_batch(half_batch, Ximg) #
        noise = np.random.uniform(-1,1, size=[half_batch, n_noise]) #

        D_loss_curr, _=sess.run([D_loss, D_solver], feed_dict={X:batch_xs, Z:noise, is_training: True})
        G_loss_curr, _=sess.run([G_loss, G_solver], feed_dict={X:batch_xs, Z:noise, is_training: True})
        losss=D_loss_curr+G_loss_curr
        print('>%d, %d/%d, d=%.3f g=%.3f' %(epoch+1, batch+1, total_batch, D_loss_curr, G_loss_curr))
    print('Epoch:', '%04d' % epoch,'D loss: {:.4}'.format(D_loss_curr),'G loss: {:.4}'.format(G_loss_curr))
    losses.append((D_loss_curr, G_loss_curr))    

    if epoch==0 or (epoch+1)%5==0:
        noise = np.random.uniform(-1.0, 1.0, size=[100, n_noise]) #
        samples = sess.run(G, feed_dict={Z: noise, is_training: False})
        save_plot(samples,epoch,5)
        save_plot(samples,epoch+1,10)
        saver.save(sess, './celeba_result/origin2_ver2_no_scale/after 100/generator_model_%03d.ckpt' % (epoch+1))
