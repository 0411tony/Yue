from data.record import Record
from tool.file import FileIO
from tool.qmath import denormalize
from tool.config import Config,LineConfig
from os.path import abspath
from time import strftime,localtime,time
from evaluation.measure import Measure
class Recommender(object):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        self.config = conf
        self.isSaveModel = False
        self.isLoadModel = False
        self.isOutput = True
        self.data = Record(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.evalConfig = LineConfig(self.config['evaluation.setup'])
        if self.evalConfig.contains('-target'):
            self.recType = self.evalConfig['-target']
        else:
            self.recType = 'track'
        if LineConfig(self.config['evaluation.setup']).contains('-cold'):
            #evaluation on cold-start users
            threshold = int(LineConfig(self.config['evaluation.setup'])['-cold'])
            removedUser = []
            for user in self.data.testSet:
                if self.data.userRecord.has_key(user) and len(self.data.userRecord[user])>threshold:
                    removedUser.append(user)
            for user in removedUser:
                del self.data.testSet[user]

        if LineConfig(self.config['evaluation.setup']).contains('-sample'):
            userList = self.data.testSet.keys()
            removedUser=userList[:int(len(userList)*0.8)]
            for user in removedUser:
                del self.data.testSet[user]

    def readConfiguration(self):
        self.algorName = self.config['recommender']
        self.output = LineConfig(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = LineConfig(self.config['item.ranking'])

    def printAlgorConfig(self):
        "show algorithm's configuration"
        print 'Algorithm:',self.config['recommender']
        print 'Training set:',abspath(self.config['record'])
        if LineConfig(self.config['evaluation.setup']).contains('-testSet'):
            print 'Test set:',abspath(LineConfig(self.config['evaluation.setup']).getOption('-testSet'))
        #print 'Count of the users in training set: ',len()
        self.data.printTrainingSize()
        print '='*80

    def initModel(self):
        pass

    def buildModel(self):
        'build the model (for model-based algorithms )'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def predict(self,user):
        return []



    def evalRanking(self):
        res = []  # used to contain the text of the result
        N = 0
        threshold = 0
        top = self.ranking['-topN'].split(',')
        top = [int(num) for num in top]
        N = int(top[-1])
        if N > 100 or N < 0:
            print 'N can not be larger than 100! It has been reassigned with 10'
            N = 10

        res.append('userId: recommendations in (itemId, ranking score) pairs, * means the item matches.\n')
        # predict
        recList = {}
        userCount = len(self.data.testSet)

        for i, user in enumerate(self.data.testSet):

            line = user + ':'
            if self.data.userRecord.has_key(user):
                predictedItems = self.predict(user)
            else:
                predictedItems = ['0']*N
            predicted = {}
            for k,item in enumerate(predictedItems):
                predicted[item] = k
            for item in self.data.userRecord[user]:
                if predicted.has_key(item[self.recType]):
                    del predicted[item[self.recType]]
            predicted = sorted(predicted.iteritems(),key=lambda d:d[1])
            predictedItems = [item[0] for item in predicted]
            recList[user] = predictedItems[:N]

            if i % 100 == 0:
                print self.algorName, self.foldInfo, 'progress:' + str(i) + '/' + str(userCount)
            for item in recList[user]:
                if self.data.testSet[user].has_key(item):
                    line += '*'
                line += item + ','

            line += '\n'
            res.append(line)
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            fileName = ''
            outDir = self.output['-dir']
            if self.ranking.contains('-topN'):
                fileName = self.config['recommender'] + '@' + currentTime + '-top-' + self.ranking['-topN']\
                           + 'items' + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, res)
            print 'The result has been output to ', abspath(outDir), '.'
        # output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['recommender'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'

        self.measure = Measure.rankingMeasure(self.data.testSet, recList,top)

        FileIO.writeFile(outDir, fileName, self.measure)
        print 'The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure))

    def execute(self):
        self.readConfiguration()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            print 'Loading model %s...' %(self.foldInfo)
            self.loadModel()
        else:
            print 'Initializing model %s...' %(self.foldInfo)
            self.initModel()
            print 'Building Model %s...' %(self.foldInfo)
            self.buildModel()

        #preict the ratings or item ranking
        print 'Predicting %s...' %(self.foldInfo)
        self.evalRanking()
        #save model
        if self.isSaveModel:
            print 'Saving model %s...' %(self.foldInfo)
            self.saveModel()

        return self.measure
