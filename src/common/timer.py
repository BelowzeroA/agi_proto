import time
import sys


class Timer:
    def __init__(self):
        self.start_time = 0
        self.global_start_time = 0
        self.global_counter = 0

    def start(self):
        self.start_time = time.time()
        if not self.global_start_time:
            self.global_start_time = self.start_time

    def show(self, text, flush=False):
        self.global_counter += 1
        line = "{} took {:5.2f} seconds".format(text, time.time() - self.start_time)
        if flush:
            sys.stdout.write("\r\x1b[K" + line.__str__())
            sys.stdout.flush()
        else:
            print(line)

    def total(self):
        return round((time.time() - self.global_start_time), 2)

    def show_average(self, text, flush=False):
        self.global_counter += 1
        line = "{} average {:5.2f} seconds".format(text, self.average())
        if flush:
            sys.stdout.write("\r\x1b[K" + line.__str__())
            sys.stdout.flush()
        else:
            print(line)

    def average(self):
        if self.global_counter == 0:
            return 0
        return round((time.time() - self.global_start_time) / self.global_counter, 2)
