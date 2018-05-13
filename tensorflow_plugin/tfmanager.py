import tensorflow as tf
import numpy as np
import tensorflow as tf
import sys, logging

def main(log_filename, lock, input_buffer, output_buffer):
    logging.basicConfig(filename=log_filename, level=logging.DEBUG)

    tfm = TFManager(lock, input_buffer, output_buffer)
    logging.info('Starting TF Session Manager')
    tfm.start_loop()

class TFManager:
    def __init__(self, lock, output_buffer, input_buffer):
        self.lock = lock
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer

    def _update(self, sess):
        print(sess.run(self.graph))

    def _build_graph(self):
        a = tf.constant(3.0, dtype=tf.float32)
        b = tf.constant(4.0)
        self.graph = a + b

    def start_loop(self):

        self._build_graph()
        logging.info('Constructed TF Model graph')

        with tf.Session() as sess:
            while True:
                self.lock.acquire()
                self._update(sess)
                self.lock.release()



