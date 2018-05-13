import tensorflow as tf
import numpy as np
import tensorflow as tf
import sys, logging

def main(log_filename, lock, input_buffer, output_buffer):
    tfm = TFManager(lock, input_buffer, output_buffer, log_filename)

    tfm.start_loop()

class TFManager:
    def __init__(self, lock, output_buffer, input_buffer, log_filename):

        self.log = logging.getLogger('tensorflow')
        fh = logging.FileHandler(log_filename)
        self.log.addHandler(fh)

        self.lock = lock
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer

        self.log.info('Starting TF Session Manager')

    def _update(self, sess):
        print(sess.run(self.graph))

    def _build_graph(self):
        a = tf.constant(3.0, dtype=tf.float32)
        b = tf.constant(4.0)
        self.graph = a + b

    def start_loop(self):

        self._build_graph()
        self.log.info('Constructed TF Model graph')

        with tf.Session() as sess:
            while True:
                break
                self.lock.acquire()
                self._update(sess)
                self.lock.release()




