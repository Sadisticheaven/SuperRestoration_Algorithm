import os

import GPUtil
from threading import Thread
import time
import psutil
from pathlib import Path

project_rootdir = 'F:/py27Proj/'
lock_file = project_rootdir + 'SuperRestoration_Algorithm/lock.txt'


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.start()
        self.gpus = GPUtil.getGPUs()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def proc_exist(process_name):
    pl = psutil.pids()
    for pid in pl:
        if psutil.Process(pid).name() == process_name:
            return pid


def lock():
    while os.path.exists(lock_file):
        print("Waiting for another main.py finishing!!")
    Path(lock_file).touch()
    # with open('lock.txt', 'w') as f:
    #     f.close()


def unlock():
    if os.path.exists(lock_file):
        os.remove(lock_file)


if __name__ == '__main__':
    # Instantiate monitor with a 10-second delay between updates
    monitor = Monitor(10)
    monitor.stop()
