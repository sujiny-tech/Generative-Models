import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image as Im
import os
import glob  
import scipy.ndimage.interpolation

def load_celeba(filename):
    data=np.load(filename)
    x=data['arr_0']
    x=x.astype('float32')
    #scale [0,255] -> [-1,1]
    x=(x-127.5)/127.5
    return x

img=load_celeba('siz84_L_50000.npz')
sample_size=img.shape[0]
input_dim=img.shape[1]
print(img.shape)
img=(img+1)/2.0


xr=np.asarray(img)/255.

nr=np.prod(xr.shape[1:3]) 
Xr=np.reshape(xr,[len(img),nr]) 

# Real image
X_rA = Xr[:1000]
# Rotated image
X_rB = Xr[1000:].reshape(-1, 84, 84)
X_rB = scipy.ndimage.interpolation.rotate(X_rB, 90, axes=(1, 2))
X_rB = X_rB.reshape(-1, 84*84)

# divide to train and test
XA=X_rA[0:500,:]
X_trainA=X_rA[500:1000,:]
XB=X_rB[0:5000,:]
X_trainB=X_rB[500:1000,:]

sample_size= X_trainA.shape[0]  
sample_size_test= XA.shape[0]
print("# train : ", sample_size)
print("# test : ", sample_size_test)
X_dim = X_trainA.shape[1]


learning_rate = 0.001
batch_size = 100
epochs =100 
h_dim = 128

#input placeholder
X_A = tf.placeholder(tf.float32, shape=[None, X_dim])
X_B = tf.placeholder(tf.float32, shape=[None, X_dim])


#D_weight, bias
D_A_W1 = tf.get_variable('D_A_W1',shape=[X_dim, h_dim],initializer=tf.contrib.layers.xavier_initializer())
D_A_b1 = tf.Variable(tf.zeros([h_dim]))
D_A_W2 = tf.get_variable('D_A_W2', shape=[h_dim, 1],initializer=tf.contrib.layers.xavier_initializer())
D_A_b2 = tf.Variable(tf.zeros([1]))

D_B_W1 = tf.get_variable('D_B_W1', shape=[X_dim, h_dim],initializer=tf.contrib.layers.xavier_initializer())
D_B_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_B_W2 = tf.get_variable('D_B_W2', shape=[h_dim, 1],initializer=tf.contrib.layers.xavier_initializer())
D_B_b2 = tf.Variable(tf.zeros([1]))

#G_weight, bias
G_AB_W1 = tf.get_variable('D_AB_W1', shape=[X_dim, h_dim],initializer=tf.contrib.layers.xavier_initializer())
G_AB_b1 = tf.Variable(tf.zeros([h_dim]))
G_AB_W2 = tf.get_variable('D_AB_W2', shape=[h_dim, X_dim],initializer=tf.contrib.layers.xavier_initializer())
G_AB_b2 = tf.Variable(tf.zeros([X_dim]))

G_BA_W1 = tf.get_variable('D_BA_W1', shape=[X_dim, h_dim],initializer=tf.contrib.layers.xavier_initializer())
G_BA_b1 = tf.Variable(tf.zeros([h_dim]))
G_BA_W2 = tf.get_variable('D_BA_W2', shape=[h_dim, X_dim],initializer=tf.contrib.layers.xavier_initializer())
G_BA_b2 = tf.Variable(tf.zeros([X_dim]))


theta_DA = [D_A_W1, D_A_W2, D_A_b1, D_A_b2]
theta_DB= [ D_B_W1, D_B_W2, D_B_b1, D_B_b2]
theta_G = [G_AB_W1, G_AB_W2, G_AB_b1, G_AB_b2,
           G_BA_W1, G_BA_W2, G_BA_b1, G_BA_b2]

#What to change later: model structure -> deep convolution network
def G_AB(X):
    h1 = tf.nn.relu(tf.matmul(X, G_AB_W1) + G_AB_b1)
    return tf.nn.sigmoid(tf.matmul(h1, G_AB_W2) + G_AB_b2)


def G_BA(X):
    h1 = tf.nn.relu(tf.matmul(X, G_BA_W1) + G_BA_b1)
    return tf.nn.sigmoid(tf.matmul(h1, G_BA_W2) + G_BA_b2)

def D_A(X):
    h1 = tf.nn.relu(tf.matmul(X, D_A_W1) + D_A_b1)
    return tf.nn.sigmoid(tf.matmul(h1, D_A_W2) + D_A_b2)


def D_B(X):
    h1 = tf.nn.relu(tf.matmul(X, D_B_W1) + D_B_b1)
    return tf.nn.sigmoid(tf.matmul(h1, D_B_W2) + D_B_b2)


X_AB = G_AB(X_A)
X_BA = G_BA(X_B)

D_A_real = D_A(X_A)
D_A_fake = D_A(X_BA)

D_B_real = D_B(X_B)
D_B_fake = D_B(X_AB)


X_ABA = G_BA(X_AB)
X_BAB = G_AB(X_BA)

#D_loss
DA_loss = 0.5 * (tf.reduce_mean((D_A_real - 1)**2)+ 0.5*tf.reduce_mean(D_A_fake**2))
DB_loss = 0.5 * (tf.reduce_mean((D_B_real - 1)**2)+ 0.5*tf.reduce_mean(D_B_fake**2))

#G_loss
GAB_loss = 0.5 * tf.reduce_mean((D_A_fake - 1)**2)+0.5 * tf.reduce_mean((D_B_fake - 1)**2)

#C_loss
C_loss=tf.reduce_mean(tf.abs(X_A-X_ABA))+tf.reduce_mean(tf.abs(X_B-X_BAB))
G_loss=GAB_loss+C_loss


# Optimizer
DA_solver = tf.train.AdamOptimizer(learning_rate).minimize(DA_loss, var_list=theta_DA)
DB_solver = tf.train.AdamOptimizer(learning_rate).minimize(DB_loss, var_list=theta_DB)
G_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=theta_G)


sess=tf.Session(); 
sess.run(tf.global_variables_initializer())
losses = []
for epoch in range(epochs):
    total_batch=int(sample_size/batch_size)
    avg_loss=0
    for ii in range(total_batch):
        if ii!=total_batch:
              XrA=X_trainA[ii*batch_size:(ii+1)*batch_size]
              XrB=X_trainB[ii*batch_size:(ii+1)*batch_size]
        else:
              XrA=X_trainA[(ii+1)*batch_size:]           
              XrB=X_trainB[(ii+1)*batch_size:]         
        DA_loss_curr,_ = sess.run([DA_loss,DA_solver], feed_dict={X_A: XrA, X_B: XrB})
        DB_loss_curr,_ = sess.run([DB_loss,DB_solver], feed_dict={X_A: XrA, X_B: XrB})
        G_loss_curr,_ = sess.run([G_loss,G_solver], feed_dict={X_A: XrA, X_B: XrB})    
        losss=DA_loss_curr+DB_loss_curr+G_loss_curr
        avg_loss+=losss/total_batch
    print('Epoch: %d' %(epoch+1),'DiscriminatorA Loss= %f,DiscriminatorB Loss= %f, Generator Loss= %f, Avg Loss=%f' %(DA_loss_curr, DB_loss_curr,G_loss_curr, avg_loss))   
    losses.append((DA_loss_curr, DB_loss_curr,G_loss_curr,avg_loss))

    
    if (epoch+1)%100==0:  
        samples_A = sess.run(X_BA, feed_dict={X_B: XB})
        samples_B = sess.run(X_AB, feed_dict={X_A: XA})
        
        #domain  A's test img
        f,axes =plt.subplots(figsize=(7,7), nrows=1, ncols=2, sharey=True, sharex=True)
        for ii in range(2):
            plt.subplot(1,2,ii+1); plt.suptitle('Domain A') 
            plt.imshow(XA[ii].reshape(84, 84),'Greys_r')
        plt.savefig('./img/Test_A_.png', bbox_inche='tight')
        
        #G_AB img          
        f,axes =plt.subplots(figsize=(7,7), nrows=1, ncols=2, sharey=True, sharex=True)      
        for ii in range(2):
            plt.subplot(1,2,ii+1); plt.suptitle('Result of G_AB') 
            plt.imshow(samples_B[ii].reshape(84, 84),'Greys_r')
        plt.savefig('./img/generated_G_AB_.png', bbox_inche='tight')
        
        #domain  B's test img
        f,axes =plt.subplots(figsize=(7,7), nrows=1, ncols=2, sharey=True, sharex=True)
        f.suptitle(epoch+1)
        f.tight_layout()
        for ii in range(2):
            plt.subplot(1,2,ii+1);plt.suptitle('Domain B') 
            plt.imshow(XB[ii].reshape(84, 84),'Greys_r')
        plt.savefig('./img/Test_B_.png', bbox_inche='tight')

        #G_BA img         
        f,axes =plt.subplots(figsize=(7,7), nrows=1, ncols=2, sharey=True, sharex=True)     
        for ii in range(2):
            plt.subplot(1,2,ii+1);plt.suptitle('Result of G_BA') 
            plt.imshow(samples_A[ii].reshape(84, 84),'Greys_r') 
        plt.savefig('./img/generated_G_BA_.png', bbox_inche='tight')
                
        
#D_loss, G_loss graph
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='DiscriminatorA')
plt.plot(losses.T[1], label='DiscriminatorB')
plt.plot(losses.T[2], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.savefig('./img/loss_graph.png', bbox_inche='tight')

#domain A img
f,axes =plt.subplots(figsize=(7,7), nrows=2, ncols=4, sharey=True, sharex=True)
f.tight_layout()
for ii in range(8):
    plt.subplot(2,4,ii+1); f.suptitle('Domain A')
    plt.imshow(X_trainA[ii].reshape(84, 84),'Greys_r')
plt.savefig('./img/A_.png', bbox_inche='tight')

#domain B img  
f,axes =plt.subplots(figsize=(7,7), nrows=2, ncols=4, sharey=True, sharex=True)
for ii in range(8):
    plt.subplot(2,4,ii+1); f.suptitle('Domain B') 
    plt.imshow(X_trainB[ii].reshape(84, 84),'Greys_r')
plt.savefig('./img/B_.png',bbox_inche='tight')
   



