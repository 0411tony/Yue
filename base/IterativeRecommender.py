from base.recommender import Recommender
from tool import config
import numpy as np
from random import shuffle
from tool.file import FileIO
from os.path import abspath
from time import strftime,localtime,time
from evaluation.measure import Measure


class IterativeRecommender(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(IterativeRecommender, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(IterativeRecommender, self).readConfiguration()
        # set the reduced dimension
        self.k = int(self.config['num.factors'])
        # set maximum iteration
        self.maxIter = int(self.config['num.max.iter'])
        # set learning rate
        learningRate = config.LineConfig(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        # regularization parameter
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU,self.regI,self.regB= float(regular['-u']),float(regular['-i']),float(regular['-b'])

    def printAlgorConfig(self):
        super(IterativeRecommender, self).printAlgorConfig()
        print 'Reduced Dimension:',self.k
        print 'Maximum Iteration:',self.maxIter
        print 'Regularization parameter: regU %.3f, regI %.3f, regB %.3f' %(self.regU,self.regI,self.regB)
        print '='*80

    def initModel(self):
        self.P = np.random.rand(self.dao.trainingSize()[0], self.k)/3 # latent user matrix
        self.Q = np.random.rand(self.dao.trainingSize()[1], self.k)/3  # latent item matrix
        self.loss, self.lastLoss = 0, 0

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def updateLearningRate(self,iter):
        if iter > 1:
            if abs(self.lastLoss) > abs(self.loss):
                self.lRate *= 1.05
            else:
                self.lRate *= 0.5

        if self.maxLRate > 0 and self.lRate > self.maxLRate:
            self.lRate = self.maxLRate


    def predict(self,u):
        return [self.dao.globalMean] * len(self.dao.item)

    def isConverged(self,iter):
        from math import isnan
        if isnan(self.loss):
            print 'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!'
            exit(-1)
        measure = self.performance()
        value = [item.strip()for item in measure]
        #with open(self.algorName+' iteration.txt')
        deltaLoss = (self.lastLoss-self.loss)
        print '%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %s %s' %(self.algorName,self.foldInfo,iter,self.loss,deltaLoss,self.lRate,measure[0][:11],measure[1][:12])
        #check if converged
        cond = abs(deltaLoss) < 1e-3
        converged = cond
        if not converged:
            self.updateLearningRate(iter)
        self.lastLoss = self.loss
        shuffle(self.dao.trainingData)
        return converged

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
        userCount = len(self.dao.testSet)
        rawRes = {}
        for i, user in enumerate(self.dao.testSet):
            itemSet = {}
            line = user + ':'
            predictedItems = self.predict(user)

            for id, score in enumerate(predictedItems):

                itemSet[self.dao.id[self.recType]] = score

            rawRes[user] = itemSet
            Nrecommendations = []
            for item in itemSet:
                if len(Nrecommendations) < N:
                    Nrecommendations.append((item, itemSet[item]))
                else:
                    break

            Nrecommendations.sort(key=lambda d: d[1], reverse=True)
            recommendations = [item[1] for item in Nrecommendations]
            resNames = [item[0] for item in Nrecommendations]

            # itemSet = sorted(itemSet.iteritems(), key=lambda d: d[1], reverse=True)
            # if bTopN:
            # find the K biggest scores
            for item in itemSet:
                ind = N
                l = 0
                r = N - 1

                if recommendations[r] < itemSet[item]:
                    while True:

                        mid = (l + r) / 2
                        if recommendations[mid] >= itemSet[item]:
                            l = mid + 1
                        elif recommendations[mid] < itemSet[item]:
                            r = mid - 1
                        else:
                            ind = mid
                            break
                        if r < l:
                            ind = r
                            break
                # ind = bisect(recommendations, itemSet[item])

                if ind < N - 1:
                    recommendations[ind + 1] = itemSet[item]
                    resNames[ind + 1] = item
            recList[user] = resNames

            if i % 100 == 0:
                print self.algorName, self.foldInfo, 'progress:' + str(i) + '/' + str(userCount)
            for item in recList[user]:
                line += item
                if self.dao.testSet[user].has_key(item[0]):
                    line += '*'

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
            self.measure = Measure.rankingMeasure(self.dao.testSet, recList, rawRes,N)

        FileIO.writeFile(outDir, fileName, self.measure)
        print 'The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure))