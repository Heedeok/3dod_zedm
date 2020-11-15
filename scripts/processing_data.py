import sys
import pyzed.sl as sl
import numpy as np

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.2f%s' % (bar, percent_done, '%'))
    sys.stdout.flush()

def main():
    print('hello')

if __name__ == '__main__':
    main()