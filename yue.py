import sys
from re import split
from tool.config import Config,LineConfig
from tool.file import FileIO
from tool.dataSplit import *
from multiprocessing import Process,Manager
from tool.file import FileIO
from time import strftime,localtime,time
from json import loads
import mkl
class Yue(object):
    def __init__(self,config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.measure = []
        self.config =config
        setup = LineConfig(config['record.setup'])
        columns = {}
        labels = setup['-columns'].split(',')
        delim = ''
        if setup.contains('-delim'):
            delim=setup['-delim']
        for col in labels:
            label = col.split(':')
            columns[label[0]] = int(label[1])
        if self.config.contains('evaluation.setup'):
            self.evaluation = LineConfig(config['evaluation.setup'])
            binarized = False
            bottom = 0
            if self.evaluation.contains('-b'):
                binarized = True
                bottom = float(self.evaluation['-b'])
            if self.evaluation.contains('-testSet'):
                #specify testSet

                self.trainingData = FileIO.loadDataSet(config['record'],columns=columns,binarized=binarized,threshold=bottom,delim=delim)
                self.testData = FileIO.loadDataSet(self.evaluation['-testSet'],binarized=binarized,columns=columns,threshold=bottom,delim=delim)

            elif self.evaluation.contains('-ap'):
                #auto partition

                self.trainingData = FileIO.loadDataSet(config['record'],columns=columns,binarized=binarized,threshold=bottom,delim=delim)
                self.trainingData,self.testData = DataSplit.\
                    dataSplit(self.trainingData,test_ratio=float(self.evaluation['-ap']))

            elif self.evaluation.contains('-byTime'):
                self.trainingData = FileIO.loadDataSet(config['record'], columns=columns, binarized=binarized,threshold=bottom, delim=delim)
                self.testData = []

            elif self.evaluation.contains('-cv'):
                #cross validation
                self.trainingData = FileIO.loadDataSet(config['record'],columns=columns,binarized=binarized,threshold=bottom,delim=delim)
                #self.trainingData,self.testData = DataSplit.crossValidation(self.trainingData,int(self.evaluation['-cv']))

        else:
            print 'Evaluation is not well configured!'
            exit(-1)

        # if config.contains('social'):
        #     self.socialConfig = LineConfig(self.config['social.setup'])
        #     self.relation = FileIO.loadRelationship(config,self.config['social'])

        print 'preprocessing...'






    def execute(self):
        #import the algorithm module
        try:
            importStr = 'from recommender.baseline.' + self.config['recommender'] + ' import ' + self.config['recommender']
            exec (importStr)
        except ImportError:
            importStr = 'from recommender.cf.' + self.config['recommender'] + ' import ' + self.config['recommender']
            try:
                exec (importStr)
            except ImportError:
                importStr = 'from recommender.advanced.' + self.config['recommender'] + ' import ' + self.config['recommender']
                exec (importStr)

        if self.evaluation.contains('-cv'):
            k = int(self.evaluation['-cv'])
            if k <= 1 or k > 10:
                k = 3
            mkl.set_num_threads(max(1,mkl.get_max_threads()/k))
            #create the manager used to communication in multiprocess
            manager = Manager()
            m = manager.dict()
            i = 1
            tasks = []

            binarized = False
            if self.evaluation.contains('-b'):
                binarized = True

            for train,test in DataSplit.crossValidation(self.trainingData,k):
                fold = '['+str(i)+']'
                # if self.config.contains('social'):
                #     recommender = self.config['recommender'] + "(self.config,train,test,self.relation,fold)"
                # else:
                recommender = self.config['recommender']+ "(self.config,train,test,fold)"
               #create the process
                p = Process(target=run,args=(m,eval(recommender),i))
                tasks.append(p)
                i+=1
            #start the processes
            for p in tasks:
                p.start()
                if not self.evaluation.contains('-p'):
                    p.join()
            #wait until all processes are completed
            if self.evaluation.contains('-p'):
                for p in tasks:
                    p.join()
            #compute the mean error of k-fold cross validation
            self.measure = [dict(m)[i] for i in range(1,k+1)]
            res = []
            for i in range(len(self.measure[0])):
                if self.measure[0][i][:3]=='Top':
                    res.append(self.measure[0][i])
                    continue
                measure = self.measure[0][i].split(':')[0]
                total = 0
                for j in range(k):
                    total += float(self.measure[j][i].split(':')[1])
                res.append(measure+':'+str(total/k)+'\n')
            #output result
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = LineConfig(self.config['output.setup'])['-dir']
            fileName = self.config['recommender'] +'@'+currentTime+'-'+str(k)+'-fold-cv' + '.txt'
            FileIO.writeFile(outDir,fileName,res)
            print 'The result of %d-fold cross validation:\n%s' %(k,''.join(res))


        else:
            # if self.config.contains('social'):
            #     recommender = self.config['recommender']+'(self.config,self.trainingData,self.testData,self.relation)'
            # else:
            recommender = self.config['recommender'] + '(self.config,self.trainingData,self.testData)'
            eval(recommender).execute()


def run(measure,algor,order):
    measure[order] = algor.execute()