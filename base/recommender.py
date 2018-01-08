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
            removedUser = {}
            for user in self.data.testSet:
                if self.data.userRecord.has_key(user) and len(self.data.userRecord[user])>threshold:
                    removedUser[user]=1
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

        N = int(self.ranking['-topN'])
        if N > 100 or N < 0:
            print 'N can not be larger than 100! It has been reassigned with 10'
            N = 10

        res.append('userId: recommendations in (itemId, ranking score) pairs, * means the item matches.\n')
        # predict
        recList = {}
        userCount = len(self.data.testSet)
        rawRes = {}
        for i, user in enumerate(self.data.testSet):
            itemSet = {}
            line = user + ':'
            predictedItems = self.predict(user)

            recList[user] = predictedItems

            if i % 100 == 0:
                print self.algorName, self.foldInfo, 'progress:' + str(i) + '/' + str(userCount)
            for item in recList[user]:
                if self.data.testSet[user].has_key(item[0]):
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
                fileName = self.config['recommender'] + '@' + currentTime + '-top-' + str(
                    N) + 'items' + self.foldInfo + '.txt'
            elif self.ranking.contains('-threshold'):
                fileName = self.config['recommender'] + '@' + currentTime + '-threshold-' + str(
                    threshold) + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, res)
            print 'The result has been output to ', abspath(outDir), '.'
        # output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['recommender'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        if self.ranking.contains('-topN'):
            self.measure = Measure.rankingMeasure(self.data.testSet, recList, rawRes,N)

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
