import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#import mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

def model(X, w_layer_1, w_layer_2, w_layer_3, dropout_input, dropout_hidden):
    X = tf.nn.dropout(X, dropout_input)

    hidden_1 = tf.nn.relu(tf.matmul(X, w_layer_1))
    hidden_1 = tf.nn.dropout(hidden_1, dropout_hidden)

    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w_layer_2))
    hidden_2 = tf.nn.dropout(hidden_2, dropout_hidden)

    return tf.matmul(hidden_2, w_layer_3)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#initilaize weigths every layer
w_layer_1 = init_weights([784, 625])
w_layer_2 = init_weights([625, 625])
w_layer_3 = init_weights([625, 10])

#define scale of dropout
dropout_input = tf.placeholder(tf.float32)
dropout_hidden= tf.placeholder(tf.float32)

#prediction output
prediction = model(X, w_layer_1, w_layer_2, w_layer_3, dropout_input, dropout_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20):
    for start, end in zip(range(0, len(mnist.train.images),100), range(100, len(mnist.train.images),100)):
        sess.run(train_op, feed_dict = {X: mnist.train.images[start:end], Y: mnist.train.labels[start:end],
                                        dropout_input: 0.8, dropout_hidden: 0.5})

print (np.mean(np.argmax(mnist.test.labels, 1) == sess.run(tf.argmax(prediction, 1),
            feed_dict = {X: mnist.test.images, Y: mnist.test.labels, dropout_input: 1.0, dropout_hidden: 1.0})))