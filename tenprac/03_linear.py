%matplotlib inline

import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# for tensorboard
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# make hypothesis
hypothesis = W * X + b

# mean ( (X*W+b - Y)^2 )
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# train with Gradient Descent method
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


# For graph
W_val = []
b_val = []
c_val = []
step_val = []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('{:>1} {:>8} {:>18} {:>13}'.format("step", "cost", "weight", "bias"))
    print()
    
    # iteration of optimization
    for step in range(200):
        # caculating
        _ , cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        
        # print        
        ### print('{:2d} {:15.6f} {:15.6f} {:15.6f}'.format(step, cost_val, sess.run(W)[0], sess.run(b)[0]))
        
        #for graph
        W_val.append(sess.run(W)[0])
        b_val.append(sess.run(b)[0])
        c_val.append(cost_val)
        step_val.append(step)

    # result
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
    #graphWriter = tf.summary.FileWriter('./graph',sess.graph)
    

    
# Graphic display
plot_cost=plt.plot(step_val, c_val, 'ro', label='cost')
plot_weight=plt.plot(step_val, W_val, 'g*', label='weight')
plot_bias=plt.plot(step_val, b_val, 'b+', label='bias')

plt.legend(loc='upper right')
plt.ylabel('Cost-Weight-bias')
plt.xlabel('Step')
plt.show()
