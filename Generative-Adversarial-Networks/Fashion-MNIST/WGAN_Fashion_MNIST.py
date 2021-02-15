# Import the requiered python packages
import math
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data 
from keras.datasets.fashion_mnist import load_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

def sample_Z(m, n):
    """ Function to generate uniform prior for G(z)
    """
    return np.random.uniform(-1., 1., size=[m, n]).astype(np.float32)

def leaky_relu(x, leak=0.2):
    return tf.maximum(x, x * leak)

def generator(z, reuse=None):
    with tf.variable_scope('generator', reuse=reuse):
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

def critic(x, reuse=None):
    with tf.variable_scope('critic', reuse=reuse):
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

def del_all_flags(FLAGS):
    """ Function to delete all flags before declare
    """
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

def runTensorFlow(batchSize):

    # Delete all Flags
    del_all_flags(tf.flags.FLAGS)

    #Set training parameters
    tf.flags.DEFINE_string('f', '', 'kernel')
    tf.flags.DEFINE_float("lr", 5e-5, "start learning rate") #0.00005  0.0002
    tf.flags.DEFINE_integer("batch_size", batchSize, "The training batch size.")
    tf.flags.DEFINE_integer("batches_per_lot", 1, "Number of batches per lot.")
    tf.flags.DEFINE_integer("num_training_steps", 10000, "The number of training steps. This counts number of lots.")
    tf.flags.DEFINE_float('weight_clip', 0.01, 'weight clipping value')

    FLAGS = tf.flags.FLAGS
    batch_size = FLAGS.batch_size


    # Initializations for a two-layer discriminator network
    Z_dim = 128
    #mnist = input_data.read_data_sets("./mnist_dataset", one_hot=True)
    #X_data, _ = mnist.train.next_batch(60000)
    (X_data, _), (_, _)=load_data()
    X_data=np.expand_dims(X_data, axis=-1)
    X_data.astype('float32')
    X_data=(X_data-127.5) / 127.5
    X_data = X_data.reshape((len(X_data), X_data.shape[1], X_data.shape[2], 1))

    X = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
    Z = tf.placeholder(tf.float32, shape=[batch_size, Z_dim])

    sample_size=X_data.shape[0] #60000/300=200
    print("sample_size : ", sample_size) #60000
    num_batches=math.floor(sample_size/batch_size)
    print("num_batches : ", num_batches) #200

    # Instantiate the Generator Network
    G_sample = generator(Z)

    # Instantiate the critic Network
    C_real, C_logit_real = critic(X)
    C_fake, C_logit_fake = critic(G_sample, reuse=True)

    # critic loss for real data
    C_loss_real = -tf.reduce_mean(C_logit_real)

    # critic loss for fake data
    C_loss_fake = tf.reduce_mean(C_logit_fake)
    C_loss = C_loss_real+C_loss_fake

    # Generator loss
    G_loss = -tf.reduce_mean(C_logit_fake)

    T_vars=tf.trainable_variables()
    critic_vars = [x for x in tf.trainable_variables() if x.name.startswith('critic')]
    generator_vars = [x for x in tf.trainable_variables() if x.name.startswith('generator')]

    # ------------------------------------------------------------------------------
    lr = tf.placeholder(tf.float32)
    # critic Optimizer
    C_solver = tf.train.RMSPropOptimizer(lr).minimize(C_loss, var_list=critic_vars) #AdamOptimizer(0.0002, 0.5)
    #C_solver = tf.train.AdamOptimizer(lr, 0.5).minimize(C_loss, var_list=critic_vars)
    clip_c = [tf.assign(var, tf.clip_by_value(var, -FLAGS.weight_clip, FLAGS.weight_clip)) for var in critic_vars]
    with tf.control_dependencies([C_solver]):
        C_solver = tf.tuple(clip_c)

    # Generator optimizer
    G_solver = tf.train.RMSPropOptimizer(lr).minimize(G_loss, var_list=generator_vars)
    #G_solver = tf.train.AdamOptimizer(lr, 0.5).minimize(G_loss, var_list=generator_vars)

    # ------------------------------------------------------------------------------
    # Set output directory
    resultDir = './nodp_wgan_mnist'
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)
   
    resultPath = resultDir + "/run_bs_{}_1".format(batch_size)

    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
        os.makedirs(resultPath+'/ckpt')

    critic_iter=5
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True

    rand_idxs = np.arange(60000)

    # Main Session
    with tf.Session(config=config) as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        saver=tf.train.Saver()#(max_to_keep=50)

        step = 0

        # Main loop
        while (step < FLAGS.num_training_steps):

            epoch = step

            #save generated image every 100 steps (any step)
            if step % 100 == 0:                    
                print("step :  " + str(step))
                n_sample = batch_size
                Z_sample = sample_Z(n_sample, Z_dim)
                
                samples = sess.run(G_sample, feed_dict={Z: Z_sample})
                samples = 0.5 * (samples + 1.0)
                 
                for k in range(64):
                    plt.subplot(8,8, k+1)
                    plt.axis('off')
                    plt.imshow(np.reshape(samples[k],(28,28)), cmap='gray_r')
                plt.savefig((resultPath + "/step_{}.png").format(str(step).zfill(3)), bbox_inches='tight')  
                plt.close()
                saver.save(sess, resultPath +'/ckpt/generator_model_%03d.ckpt' % (step+1))

            idx = 0
            while idx<(num_batches):
                critic_i=0
                while critic_i < critic_iter and idx<num_batches :
                    batch_idxs = rand_idxs[idx*batch_size: (idx+1)*batch_size] 
                    X_mb = X_data[batch_idxs, :]
                    #X_mb = 2 * X_mb.astype(np.float32) - 1.0  
                    #X_mb = X_mb.reshape(-1, 28, 28, 1)
                    Z_sample = sample_Z(batch_size, Z_dim)

                    # Update the discriminator network
                    _, C_loss_real_curr, C_loss_fake_curr = sess.run([C_solver, C_loss_real, C_loss_fake], feed_dict={X: X_mb, Z: Z_sample, lr: FLAGS.lr})
                    critic_i+=1
                    idx+=1 
                # Update the generator network
                _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, lr: FLAGS.lr}) #, lr: FLAGS.lr 
            print("step :  " + str(step))
            print("c_loss: ", C_loss_real_curr+C_loss_fake_curr)
            print("g_loss: ", G_loss_curr)
            step = step + 1
          
        n_sample = batch_size
        Z_sample = sample_Z(n_sample, Z_dim)
                    
        samples = sess.run(G_sample, feed_dict={Z: Z_sample})

        ########################
        for k in range(64):
            plt.subplot(8,8, k+1)
            plt.axis('off')
            plt.imshow(np.reshape(samples[k],(28,28)), cmap='gray_r')
        plt.savefig((resultPath + "/Final_step_{}.png").format(str(step).zfill(3)), bbox_inches='tight')
        plt.close()
        saver.save(sess, resultPath +'/ckpt/generator_model_%03d.ckpt' % (step+1))
        ########################

batchSizeList = [300]

for batchSize in batchSizeList:
    print("iRunning TensorFlow with batchSize=%d\n" % (batchSize))
    runTensorFlow(batchSize)

