import numpy as np

class SymmetricMatrix(object):
    def __init__(self, shape):
        self.symMatrix = {}
        self.shape = (shape,shape)

    def __getitem__(self, item):
        if item in self.symMatrix:
            return self.symMatrix[item]
        return {}

    def set(self,i,j,val):
        if i not in self.symMatrix:
            self.symMatrix[i] = {}
        self.symMatrix[i][j]=val
        if j not in self.symMatrix:
            self.symMatrix[j] = {}
        self.symMatrix[j][i] = val


    def get(self,i,j):
        if i not in self.symMatrix or j not in self.symMatrix[i]:
            return 0
        return self.symMatrix[i][j]

    def contains(self,i,j):
        if i in self.symMatrix and j in self.symMatrix[i]:
            return True
        else:
            return False

