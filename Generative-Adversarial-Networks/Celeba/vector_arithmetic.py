import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        #v = slerp(ratio, p1, p2)
        vectors.append(v)
    return np.asarray(vectors)

#https://en.wikipedia.org/wiki/Slerp
#spherical linear interpolation (slerp)
def slerp(val, low, high):
    step1=np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high))) #to compute omega
    omega=np.arccos(step1, -1, 1)
    so=np.sin(omega)
    if so==0: #lerp
        return (1.0-val)*low+val*high
    return np.sin((1.0-val)*omega)/so*low+np.sin(val*omega)/so*high

def save_plot(examples, epoch, n=5):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    plt.savefig('./celeba_result/origin2/{}.png'.format(str(epoch).zfill(4)),bbox_inches='tight')
    plt.close()

def plot_generated(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :])
    plt.savefig('./celeba_result/origin2/latent_points_{}.png'.format(examples),bbox_inches='tight')
    plt.title("interpolate_DCGAN")
    plt.show()

#average list of latent space vectors
def average_points(points, ix):
    index=[i-1 for i in ix] #[a, b, c]->[a-1, b-1, c-1]
    p1=points[index]
    avg_point=np.mean(p1, axis=0)
    all_point=np.vstack((p1, avg_point))
    return all_point

#####
'''
is_training = tf.placeholder(tf.bool)
 
def leaky_relu(x, leak=0.2):
    return tf.maximum(x, x * leak)
 
def generator(noise, reuse=None):
    with tf.variable_scope('generator', reuse=reuse):
        output = tf.layers.dense(noise, 128*5*5)
        output = leaky_relu(output)
        output = tf.reshape(output, [-1, 5,5,128])

        output = tf.layers.conv2d_transpose(output, 128, [4,4], strides=(2,2), padding='SAME')
        output = leaky_relu(output)

        output = tf.layers.conv2d_transpose(output, 128, [4,4], strides=(2,2), padding='SAME')
        output = leaky_relu(output)

        output = tf.layers.conv2d_transpose(output, 128, [4,4], strides=(2,2), padding='SAME')
        output = leaky_relu(output)
        
        output = tf.layers.conv2d_transpose(output, 128, [4,4], strides=(2,2), padding='SAME')
        output = leaky_relu(output)

        output = tf.layers.conv2d(output, 3, [5,5], padding='SAME')
        output = tf.tanh(output)
    return output

G = generator(Z)
'''

def vector_arithmetic():
    noise=np.random.uniform(-1, 1, size=[100, 100])
    np.savez_compressed('latent_points.npz', noise)

    #retrieve specific points
    smiling_woman_ix=[92, 98, 99]
    neutral_woman_ix=[9, 21, 79]
    neutral_man_ix=[10, 30, 45]

    data=np.load('latent_points.npz') #noise np
    points=data['arr_0']

    smiling_woman=average_points(noise, smiling_woman_ix) #original point + average_points
    neutral_woman=average_points(noise, neutral_woman_ix)
    neutral_man=average_points(noise, neutral_man_ix)

    all_points=np.vstack((smiling_woman, neutral_woman, neutral_man)) #all points
    
    result_=smiling_woman[-1] - neutral_woman[-1] + netral_man[-1]
    result_=np.expand_dims(result_, 0)

    return all_points, result_


saver=tf.train.Saver()

sess=tf.Session()
sess.run(tf.global_variables_initializer())

save_file='./celeba_result/origin2/generator_model_100.ckpt'
saver.restore(sess, save_file)

'''
noise=np.random.uniform(-1, 1, size=[100, 100])

#np.savez_compressed('latent_points.npz', noise)

#retrieve specific points
smiling_woman_ix=[92, 98, 99]
neutral_woman_ix=[9, 21, 79]
neutral_man_ix=[10, 30, 45]

#data=np.load('latent_points.npz') #noise np
#points=data['arr_0']
smiling_woman=average_points(noise, smiling_woman_ix) #original point + average_points
neutral_woman=average_points(noise, neutral_woman_ix)
neutral_man=average_points(noise, neutral_man_ix)

all_points=np.vstack((smiling_woman, neutral_woman, neutral_man)) #all points
'''

all_points, result_=vector_arithmetic()

x=sess.run(G, feed_dict={Z:all_points, is_training:False})
x=(x+1)/2.0 #[0,1] data scale
plot_generated(x, 3)

y=sess.run(G, feed_dict={Z:result_, is_training:False})
y=(x+1)/2.0
plot_generated(y, 3)