import tensorflow as tf

with tf.variable_scope("xy"):
    var0 = tf.get_variable('ori_var0', [1,1])
    var0 = tf.get_variable('copy_var0', [1,1])
    var1 = tf.get_variable('ori_var1', [1,1])
    var1 = tf.get_variable('copy_var1', [1,1])


vars = [var for var in tf.trainable_variables() if "copy" in var.name]
#
vars1 = [var for var in tf.trainable_variables()]

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)
# print sess.run(var)
# print sess.run(var2)
#
# print sess.run(var2)
# print sess.run(var)

for item in vars1:
    print item.name
print len(vars)