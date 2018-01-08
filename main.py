import sys
sys.path.append("..")
import yue
from tool.config import Config




if __name__ == '__main__':

    print '='*80
    print '   Yue: Library for Music Recommendation.   '
    print '='*80
    print 'CF-based Recommenders:'
    print '1. BPR'


    print 'Content-based Recommenders:'



    print 'Hybrid Recommenders:'


    print 'Baselines:'
    print 'b1. MostPop   b2. Rand'
    print '='*80
    algor = -1
    conf = -1
    order = raw_input('Please enter the num of the algorithm to run it:')
    import time
    s = time.time()
    if order=='1':
        conf = Config('./config/BPR.conf')

    elif order == 'b1':
        conf = Config('./config/MostPop.conf')

    elif order == 'b2':
        conf = Config('./config/rand.conf')

    else:
        print 'Error num!'
        exit(-1)
    musicSys = yue.Yue(conf)
    musicSys.execute()
    e = time.time()
    print "Run time: %f s" % (e - s)
