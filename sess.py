# session inside  function is another session
import tensorflow as tf

def run_b():
    a = tf.Variable([2.0])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(a))
        print(sess)
        
a = tf.Variable([1.0])
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess)
    run_b()

