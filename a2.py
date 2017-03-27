import tensorflow as tf
i0 = tf.constant(0)
m0=tf.ones([1,2])
m1=m0
c=lambda i, m , m1: i<3
b=lambda i, m , m1:[i+1,m, m1+m]

d = tf.while_loop(
    c,b,loop_vars=[i0,m0,m1]
)
# shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])]
sess = tf.Session()
sess.run(tf.global_variables_initializer())
dd= sess.run(d)
print dd