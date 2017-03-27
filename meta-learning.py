

import numpy as np
import tensorflow as tf
import math


from tf_ops_meta import *
import random

#####################################


paras = []
paras.append([0.6, 1.2,0.5])
paras.append([1, 2,1.5])
paras.append([1.5, 2.5,2])
paras.append([1.1, -1,-1])
paras.append([2, 0.5,2])
paras.append([2.5, 1,0])
paras.append([1.1, 2,3])

NSAMPLE = 250
batch_size = 32
data_list=[]

for i in range(len(paras)):
    x_data = np.float32(np.random.uniform(-5, 5, (1, NSAMPLE))).T
    y_data = np.float32(np.sin(paras[i][0] * x_data) * paras[i][1] + paras[i][2])
    data_list.append(zip(x_data,y_data))


#####################################

K=3
META=6
alpha=0.001
beta = alpha * 0.6 / META


x = tf.placeholder(tf.float32, [None, 1] ,'x')
y = tf.placeholder(tf.float32, [None, 1] ,'y')
# step = tf.placeholder(tf.int32, name='step')
is_training = tf.placeholder(tf.bool, name='is_train')

x_split = tf.split(x,[batch_size,batch_size,batch_size,batch_size,batch_size])

#####################################

with tf.variable_scope("xy"):
    input=x
    target=y
    input=lrelu(fc(input,12,'fc1'))
    input=lrelu(fc(input,6,'fc2'))
    output=fc(input,1,'fc3')

    l1_loss=tf.reduce_mean(tf.abs(output-target))
    optimizer_target = tf.train.RMSPropOptimizer(alpha, decay=0.95, momentum=0.9, epsilon=1e-8)
    optimizer_global = tf.train.RMSPropOptimizer(beta, decay=0.95, momentum=0.9, epsilon=1e-8)
sess = tf.Session()


vars_init = [var for var in tf.trainable_variables() if var.name.startswith("xy") and 'init' in var.name]
vars_copy = [var for var in tf.trainable_variables() if var.name.startswith("xy") and 'copy' in var.name]

gradient_all = optimizer_target.compute_gradients(l1_loss)

grads_vars = [v for (g,v) in gradient_all if g is not None]
gradient = optimizer_target.compute_gradients(l1_loss, vars_init)

grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v) for (g,v) in gradient]

train_op_target = optimizer_target.apply_gradients(gradient)
train_op_global = optimizer_global.apply_gradients(grads_holder)


sess.run(tf.global_variables_initializer())




#####################################

def DealGradients(gradient_results):
    gradient = gradient_results[0]

    for i in range(1,len(gradient_results)):
        for j in range(len(gradient)):
            gradient[j] += gradient_results[i][j]

    dict = {}
    for i in range(len(gradient)):
        k = grads_holder[i][0]
        dict[k] = gradient[i][0]
    return dict


#####################################

def forward_one_step():
    gradient_results = []
    forward_loss = 0
    for meta in range(META):
        copy_paras(vars_init, vars_copy, sess)

        for k in range(K):
            data_feed = random.sample(data_list[meta], batch_size)
            x_feed, y_feed = zip(*data_feed)
            _ = sess.run(train_op_target, feed_dict={x: x_feed, y: y_feed, is_training: True})

        loss, gradient_result = sess.run([l1_loss, gradient], feed_dict={x: x_feed, y: y_feed, is_training: True})
        forward_loss += loss

        gradient_results.append(gradient_result)
        copy_paras(vars_copy, vars_init, sess)

    grads_dict = DealGradients(gradient_results)
    _ = sess.run(train_op_global,feed_dict=grads_dict)
    print "step_loss: " + str(forward_loss)

step=0
while(True):

    forward_one_step()
    print "step " + str(step)


print "down..."
# ll = sess.run(glist, feed_dict={x: x_feed, y: y_feed, step:0, is_training: True})

# print len(grad_run)
# for item in grad_run:
#     print("########")
#
#     print np.shape(item)