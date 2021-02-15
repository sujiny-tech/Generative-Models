import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import glob

def save_plot(examples, epoch, n=5):
    # scale from [-1,1] to [0,1]
    #examples = (examples + 1) / 2.0  #####!!!@!!!
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    plt.savefig('./celeba_result/origin2_ver2_no_scale/{}.png'.format(str(epoch).zfill(4)),bbox_inches='tight')
    plt.close()

# uniform interpolation between two points in latent space !!!
def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        #v = (1.0 - ratio) * p1 + ratio * p2
        v = slerp(ratio, p1, p2)
        vectors.append(v)
    return asarray(vectors)

def plot_generated(examples, a,b, name):
    # plot images
    for i in range(a * b):
        # define subplot
        plt.subplot(a, b, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :])
    plt.savefig('./celeba_result/origin2_ver2_no_scale/{}.png'.format(str(name)),bbox_inches='tight')

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
    step1=np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1) #to compute omega
    omega=np.arccos(step1)
    so=np.sin(omega)
    if so==0: #lerp
        return (1.0-val)*low+val*high
    return np.sin((1.0-val)*omega)/so*low+np.sin(val*omega)/so*high

#average list of latent space vectors
def average_points(points, ix):
    index=[i-1 for i in ix] #[a, b, c]->[a-1, b-1, c-1]
    p1=points[index]
    avg_point=np.mean(p1, axis=0)
    all_point=np.vstack((p1, avg_point))
    return all_point

def smiling_man(noise):
    smiling_woman_ix=[12, 55, 70]
    neutral_woman_ix=[4,9,60]
    neutral_man_ix=[1, 21, 99]#8->21, 41->99, 66

    smiling_woman=average_points(noise, smiling_woman_ix) #original point + average_points
    neutral_woman=average_points(noise, neutral_woman_ix)
    neutral_man=average_points(noise, neutral_man_ix)

    all_points=np.vstack((smiling_woman, neutral_woman, neutral_man)) #all points
    
    result_=smiling_woman[-1] - neutral_woman[-1] + neutral_man[-1]########
    result_=np.expand_dims(result_, 0)

    result_1=smiling_woman[0] - neutral_woman[0] + neutral_man[0]########
    result_1=np.expand_dims(result_1, 0)

    result_2=smiling_woman[1] - neutral_woman[1] + neutral_man[1]########
    result_2=np.expand_dims(result_2, 0)

    result_3=smiling_woman[2] - neutral_woman[2] + neutral_man[2]########
    result_3=np.expand_dims(result_3, 0)

    return all_points, result_, result_1, result_2, result_3

def glasses_woman(noise):
    glasses_man_ix=[42, 34, 91]#20->30, 34 /42->76
    no_glaases_man_ix=[1, 44, 77]#41->44
    no_glaases_woman_ix=[36,47,58]

    glasses_man=average_points(noise, glasses_man_ix) #original point + average_points
    no_glaases_man=average_points(noise, no_glaases_man_ix)
    no_glaases_woman=average_points(noise, no_glaases_woman_ix)

    all_points=np.vstack((glasses_man, no_glaases_man, no_glaases_woman)) #all points
    
    result_=glasses_man[-1] - no_glaases_man[-1] + no_glaases_woman[-1]########
    result_=np.expand_dims(result_, 0)

    result_1=glasses_man[0] - no_glaases_man[0] + no_glaases_woman[0]########
    result_1=np.expand_dims(result_1, 0)

    result_2=glasses_man[1] - no_glaases_man[1] + no_glaases_woman[1]########
    result_2=np.expand_dims(result_2, 0)

    result_3=glasses_man[2] - no_glaases_man[2] + no_glaases_woman[2]########
    result_3=np.expand_dims(result_3, 0)

    return all_points, result_, result_1, result_2, result_3

##
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


############
sess=tf.Session()
sess.run(tf.global_variables_initializer())

saver=tf.train.Saver()

new_saver=tf.train.import_meta_graph('./celeba_result/origin2_ver2_no_scale/after 100/generator_model_185.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./celeba_result/origin2_ver2_no_scale/after 100/'))

######vectore arithmetic###########
#noise=np.random.uniform(-1, 1, size=[100, 100])
#np.savez_compressed('latent_points.npz', noise)

data=np.load('latent_points.npz') #noise np
noise=data['arr_0']

n=20
results=None
for j in range(0, n, 2):
    interpolated=interpolate_points(noise[j], noise[j+1])
    x=sess.run(G, feed_dict={Z:interpolated, is_training:False})
    if results is None:
        results=x
    else:
        results=np.vstack((results, x))
plot_generated(results, 10, 10, 'interpolated_image')
plt.close()

#writer=tf.summary.FileWriter('./celeba_result/origin2_ver2_no_scale/logs', sess.graph)

x=sess.run(G, feed_dict={Z:noise, is_training:False})

save_plot(x, 185, 10)

#smiling_man = smiling_woman - neutral_woman + neutral_man
all_points, result_, result_1, result_2, result_3=smiling_man(noise)
x=sess.run(G, feed_dict={Z:all_points, is_training:False})
plot_generated(x, 3,4, 'smiling_woman_neutral_woman_man' )
plt.close()

#y = average_smiling_man   /  y1, y2, y3 = each arithmetic result
y=sess.run(G, feed_dict={Z:result_, is_training:False})
y1=sess.run(G, feed_dict={Z:result_1, is_training:False})
y2=sess.run(G, feed_dict={Z:result_2, is_training:False})
y3=sess.run(G, feed_dict={Z:result_3, is_training:False})

all_=np.vstack((y1, y2, y3, y))
plot_generated(all_, 1, 4, 'all_smiling')
plt.close()

plt.imshow(y[0])
plt.savefig('./celeba_result/origin2_ver2_no_scale/smiling_man.png',bbox_inches='tight')
plt.close()

#glasses_woman = glasses_man - no_glasses_man + no_glasses_woman
all_pts, res_, res_1, res_2, res_3=glasses_woman(noise)
x1=sess.run(G, feed_dict={Z:all_pts, is_training:False})
plot_generated(x1, 3, 4, 'glasses_man_neutral_man_neutral_woman')
plt.close()

#z1 = average_glasses_woman   /  z2, z3, z4 = each arithmetic result
z1=sess.run(G, feed_dict={Z:res_, is_training:False})
z2=sess.run(G, feed_dict={Z:res_1, is_training:False})
z3=sess.run(G, feed_dict={Z:res_2, is_training:False})
z4=sess.run(G, feed_dict={Z:res_3, is_training:False})

all_1=np.vstack((z2, z3, z4, z1))
plot_generated(all_1, 1, 4, 'all_glasses')
plt.close()

y1=sess.run(G, feed_dict={Z:res_, is_training:False})
plt.imshow(y1[0])
plt.savefig('./celeba_result/origin2_ver2_no_scale/glasses_woman.png',bbox_inches='tight')
plt.close()

#writer=tf.summary.FileWriter('./celeba_result/origin2_ver2_no_scale/logs')

'''
for i in range(105, 170, 5):
    ###load train model
    new_saver=tf.train.import_meta_graph('./celeba_result/origin2_ver2_no_scale/after 100/generator_model_{}.ckpt.meta'.format(str(i)))
    new_saver.restore(sess, tf.train.latest_checkpoint('./celeba_result/origin2_ver2_no_scale/after 100/'))
    
    data=np.load('latent_points.npz') #noise np
    noise=data['arr_0']
    
    x=sess.run(G, feed_dict={Z:noise, is_training:False})
    save_plot(x, i, 10)
    plt.close()
'''
