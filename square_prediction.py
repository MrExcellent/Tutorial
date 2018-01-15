import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#add a nervous layer
def layers(inputs,in_num,out_num,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_num,out_num]))
    biases = tf.Variable(tf.zeros([1,out_num])+0.1)
    y = tf.matmul(inputs,Weights) + biases

    if activation_function is None:
        outputs = y
    else:
        outputs = activation_function(y)
    return outputs

x_data = np.linspace(-1,1,400)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 + noise

xt = tf.placeholder(tf.float32,[None,1])
yt= tf.placeholder(tf.float32,[None,1])

hidden_layer = layers(xt,1,8,activation_function=tf.nn.sigmoid)
prediction = layers(hidden_layer,8,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(yt-prediction),reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
Sess = tf.Session()
Sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(5000):
    Sess.run(train,feed_dict={xt:x_data,yt:y_data})
    if i % 50 == 0:
        try:
           ax.lines.remove(lines[0])
        except Exception:
           pass
        prediction_value = Sess.run(prediction,feed_dict={xt:x_data})
        lines = ax.plot(x_data,prediction_value,"r-",lw=3)

        plt.pause(0.5)
