import sys
sys.path.append("..")
import yue
from tool.config import Config
import time

if __name__ == '__main__':

    print ('='*80)
    print ('   Yue: Library for Music Recommendation.   ')
    print ('='*80)
    print ('CF-based Recommenders:')
    print ('1. BPR   2. FISM   3. WRMF   4. IPF')
    print ('5. UserKNN')

    print ('Content-based Recommenders:\n')

    print ('Hybrid Recommenders:\n')

    print ('Advanced Recommenders:')
    print ('a1. CUNE   a2. Song2vec   a3. BasicMF')
    print ('a4. CDAE   a5. DMF')

    s = time.time()
    
    print ('Baselines:')
    print ('b1. MostPop   b2. Rand')
    print ('='*80)
    algor = -1
    conf = -1
    order = input('Please enter the num of the algorithm to run it:')

    import time
    s = time.time()

    if order=='1':
        conf = Config('./config/BPR.conf')

    elif order=='2':
        conf = Config('./config/FISM.conf')

    elif order=='3':
        conf = Config('./config/WRMF.conf')

    elif order=='4':
        conf = Config('./config/IPF.conf')

    elif order=='5':
        conf = Config('./config/UserKNN.conf')

    elif order == 'b1':
        conf = Config('./config/MostPop.conf')

    elif order == 'b2':
        conf = Config('./config/rand.conf')


    elif order == 'a1':
        conf = Config('./config/CUNE.conf')

    elif order == 'a2':
        conf = Config('./config/Song2vec.conf')
        
    elif order == 'a3':
        conf = Config('./config/BasicMF.conf')
    
    elif order == 'a4':
        conf = Config('./config/CDAE.conf')

    elif order == 'a5':
        conf = Config('./config/DMF.conf')

    else:
        print ('Error num!')
        exit(-1)

    musicSys = yue.Yue(conf)
    musicSys.execute()
    e = time.time()
    print ("Run time: %f s" % (e - s))
    
