import _spinlock as lock
import sys, time, multiprocessing

def master(lock, sleeps=0.05):
    for i in range(10):
        print('Going to cede to slave. Iteration:', i)
        sys.stdout.flush()
        time.sleep(sleeps)
        lock.await()
        print('Masster is awake now')
        sys.stdout.flush()

def slave(lock, sleeps=0.2):
    for i in range(10):
        lock.start()
        print('Worker is doing important work', i)
        sys.stdout.flush()
        time.sleep(sleeps)
        print('Worker is done', i)
        sys.stdout.flush()
        lock.end()
        print('Worker has ended', i)

if __name__ == '__main__':
    lock = lock.TaskLock()
    p = multiprocessing.Process(target=slave, args=(lock,))
    p.start()
    master(lock)