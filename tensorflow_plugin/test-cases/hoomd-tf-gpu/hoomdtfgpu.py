import hoomd
import multiprocessing

def start_tf():
    import tensorflow as tf
    g = tf.zeros(10)
    p = tf.Print(g, [g])
    with tf.Session() as sess:
        sess.run(p)


p = multiprocessing.Process(target=start_tf)
p.start()
hoomd.context.initialize()



