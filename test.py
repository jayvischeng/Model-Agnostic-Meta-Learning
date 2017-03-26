import tensorflow as tf

with tf.variable_scope("xy"):
    var0 = tf.get_variable('init_var0', [1,1])
    var0_cp = tf.get_variable('copy_var0', [1,1])
    var1 = tf.get_variable('init_var1', [1,1])
    var1_cp = tf.get_variable('copy_var1', [1,1])
#var2 = tf.get_variable('var2', [3,2])


vars_init = [var for var in tf.trainable_variables() if var.name.startswith("xy") and 'init' in var.name]
vars_copy = [var for var in tf.trainable_variables() if var.name.startswith("xy") and 'copy' in var.name]


def copy_paras(frm, to):
    list = []
    for i in range(len(frm)):
        ass = to[i].assign(frm[i])
        list.append(ass)
    sess.run(list)

op_add = var0.assign(var1)
# op_ = var1.assign(var0)


#copy_first_variable = var2.assign(var)
init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)


copy(init=vars_init, copy=vars_copy)

for item in vars_init:
    print item.name
    print sess.run(item)


print '-----'
sess.run(op_add)

for item in vars_init:
    print sess.run(item)

print '-----'
copy(init=vars_copy, copy=vars_init)

for item in vars_init:
    print sess.run(item)