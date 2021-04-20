import logging
import sys

def dummy_test():
    logging.basicConfig(level = logging.INFO, stream = sys.stdout)
    log = logging.getLogger('timely test')
    log.setLevel(logging.INFO)
    print('normal print 1')
    log.warning(' Just a test warning message. Ignore it.')
    logging.warning('This is how to print a warning')
    log.info('This is how to print an info, but pytest will not display it by default')
    print('normal print 2')
    assert False, "dummy assertion"


