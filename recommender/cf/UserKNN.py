from base.recommender import Recommender
from tool import qmath
from structure.symmetricMatrix import SymmetricMatrix
from collections import defaultdict

class UserKNN(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(UserKNN, self).__init__(conf,trainingSet,testSet,fold)
        self.userSim = SymmetricMatrix(len(self.data.name2id['user']))
        self.topUsers = {}

    def readConfiguration(self):
        super(UserKNN, self).readConfiguration()
        self.neighbors = int(self.config['num.neighbors'])

    def printAlgorConfig(self):
        "show algorithm's configuration"
        super(UserKNN, self).printAlgorConfig()
        print 'Specified Arguments of',self.config['recommender']+':'
        print 'num.neighbors:',self.config['num.neighbors']
        print '='*80

    def initModel(self):
        self.computeCorr()

    def predict(self,u):
        recommendations = []
        for item in self.data.listened[self.recType]:
            sum, denom = 0, 0
            for simUser in self.topUsers[u]:
                #if user n has rating on item i
                    if self.data.listened[self.recType][item].has_key(simUser[0]):
                        similarity = simUser[1]
                        score = self.data.listened[self.recType][item][simUser[0]]
                        sum += similarity*score
                        denom += similarity
            if sum!=0:
                score = sum / float(denom)
                recommendations.append((item,score))
        recommendations = sorted(recommendations,key=lambda d:d[1],reverse=True)
        recommendations = [item[0] for item in recommendations]
        return recommendations


    def computeCorr(self):
        'compute correlation among users'
        userListen = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                if userListen[user].has_key(item[self.recType]):
                    userListen[user][item[self.recType]] += 1
                else:
                    userListen[user][item[self.recType]] = 0
        print 'Computing user similarities...'
        for ind,u1 in enumerate(userListen):
            set1 = set(userListen[u1].keys())
            for u2 in userListen:
                if u1 <> u2:
                    if self.userSim.contains(u1,u2):
                        continue
                    set2 = set(userListen[u2].keys())
                    sim = self.jaccard(set1,set2)
                    self.userSim.set(u1,u2,sim)
            self.topUsers[u1] = sorted(self.userSim[u1].iteritems(), key=lambda d: d[1], reverse=True)[:self.neighbors]
            if ind%100==0:
                print ind,'/',len(userListen), 'finished.'
        print 'The user correlation has been figured out.'

    def jaccard(self,s1,s2):
        return 2*len(s1.intersection(s2))/(len(s1.union(s2))+0.0)

