import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#import MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#add a nervous layer
def add_layer(inputs,in_num,out_num,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_num,out_num]))
    biases = tf.Variable(tf.zeros([1,out_num])+0.1)
    y = tf.matmul(inputs,Weights) + biases

    if activation_function is None:
        outputs = y
    else:
        outputs = activation_function(y)
    return outputs

xt = tf.placeholder(tf.float32,[None,784])
yt = tf.placeholder(tf.float32,[None,10])

hidden_layer1 = add_layer(xt,784,38,activation_function=tf.nn.sigmoid)
hidden_layer2 = add_layer(hidden_layer1,38,38,activation_function=tf.nn.sigmoid)
output = add_layer(hidden_layer2,38,10,activation_function=None)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yt, logits=output))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


Sess = tf.Session()
Sess.run(tf.global_variables_initializer())

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    Sess.run(train,feed_dict={xt:batch[0],yt:batch[1]})

correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(yt,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(Sess.run(accuracy,feed_dict={xt: mnist.test.images, yt: mnist.test.labels}))