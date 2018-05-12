import tensorflow as tf

def main(lock, input_buffer, output_buffer):
    tfm = TFManager(lock, input_buffer, output_buffer)


class TFManager:
    def __init__(self, lock, output_buffer, input_buffer):
        self.lock = lock
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer

        self.lock.acquire()

