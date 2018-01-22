import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#import MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


traindata,trainlabel=mnist.train.next_batch(2000)
testdata,testlabel=mnist.test.next_batch(100)

#creat nodes
train_tensor=tf.placeholder('float',[None,784])
test_tensor=tf.placeholder('float',[784])

#calculate euclidena distance
Euclidean_distance = tf.reduce_sum(tf.abs(tf.add(train_tensor,tf.negative(test_tensor))),reduction_indices=1)
pred = tf.argmin(Euclidean_distance,0)

Sess=tf.Session()
Sess.run(tf.global_variables_initializer())

accuracy = 0

for i in range(50):
    index=Sess.run(pred,feed_dict={train_tensor:traindata,test_tensor:testdata[i]})
    print('test No.%d,the real label %d, the predict label %d'%(i,np.argmax(testlabel[i]),np.argmax(trainlabel[index])))
    if np.argmax(testlabel[i])==np.argmax(trainlabel[index]):
        accuracy+=1
print("result:%f"%(1.0*accuracy/50))