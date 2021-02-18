import random
import statistics
import math

"""
###########################################SETTINGS#######################################
"""
K = 60 #60
alpha = 0.005 #0.01
beta = 0.01
numIter = 30 #20
"""
###########################################################################################
"""
class Io:
    def zeroElements(self, mat):
        xs = []
        ys = []
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] == 0:
                    xs.append(i)
                    ys.append(j)
        return xs, ys

    def input(self):
        np = Numpybyhand()
        configLine = input()
        numRatings = int(configLine.split('\t')[0])
        numUsers = int(configLine.split('\t')[1])
        numBooks = int(configLine.split('\t')[2])

        Rin = np.mat(numUsers, numBooks, 0)

        for i in range(numRatings):
            dataLine = input()
            n = int(dataLine.split('\t')[0])
            m = int(dataLine.split('\t')[1])
            val = int(dataLine.split('\t')[2])
            Rin[n][m] = val

        return Rin
    
    def sortResults(self, Rin, Rout):
        xs, ys = self.zeroElements(Rin)
        Rd = [[] for i in range(max(xs) + 1)]
        Rk = [[] for i in range(max(xs) + 1)]
        for i in range(len(Rd)):
            itemDict = {}
            for x, y in zip(xs, ys):
                if x == i:
                    itemDict[Rout[x][y]] = y
            Rd[i].append(itemDict)
        for i in range(len(Rd)):
            Rk[i] = sorted(Rd[i][0], reverse=True)
        return Rd, Rk

    def printResult(self, m):
        for row in m:
            j = 0
            for i in row:
                print('%f' % (i), end='')
                if j < len(row) - 1:
                    print('\t', end='')
                    j = j + 1
            print("\n", end='')

    def printDict(self, Rd, Rk):
        for n in range(len(Rk)):
            j = 0
            for m in range(len(Rk[n])):
                if m < 10:
                    print('%d'%(Rd[n][0][Rk[n][m]]), end='')
                    if j < 9:
                        print('\t', end='')
                        j = j + 1
            print("\n", end='')


class Numpybyhand:
    def mat(self, n, m, fillwith):
        matrix = [[fillwith for x in range(m)] for y in range(n)]
        return matrix

    def randomise(self, mat, mu, sigma):
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                mat[i][j] = random.normalvariate(mu, sigma)
        return mat

    def transpose(self, mat):
        return list(map(list, zip(*mat)))

    def nonZeroElements(self, mat):
        m = []
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] != 0:
                    m.append(mat[i][j])
        return m

    def nonZeroIndexes(self, mat):
        xs = []
        ys = []
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] != 0:
                    xs.append(i)
                    ys.append(j)
        return xs, ys

    def zeroElements(self, mat):
        xs = []
        ys = []
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] == 0:
                    xs.append(i)
                    ys.append(j)
        return xs, ys


    def add(self, mat1, mat2):
        res = [x1 + x2 for (x1, x2) in zip(mat1, mat2)]
        return res

    def substract(self, mat1, mat2):
        res = [x1 - x2 for (x1, x2) in zip(mat1, mat2)]
        return res

    def multiply(self, mat1, mat2):
        res = [x * mat2 for x in mat1]
        return res

    def dot(self, mat1, mat2):
        res = [x1 * x2 for (x1, x2) in zip(mat1, mat2)]
        return sum(res)


class MF():
    def __init__(self, R, K, alpha, beta, iterations):
        self.np = Numpybyhand()
        self.R = R
        self.numUsers, self.numItems = len(R), len(R[0])
        self.K = K
        self.P = self.np.mat(self.numUsers, self.K, 0)
        self.Q = self.np.mat(self.numItems, self.K, 0)
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        self.P = self.np.mat(self.numUsers, self.K, 0)
        self.P = self.np.randomise(self.P, 0, 1./self.K)
        self.Q = self.np.mat(self.numItems, self.K, 0)
        self.Q = self.np.randomise(self.P, 0, 1./self.K)

        self.bu = [0 for i in range(self.numUsers)]
        self.bi = [0 for i in range(self.numItems)]
        self.b = statistics.mean(self.np.nonZeroElements(self.R))

        self.samples = [
            (i, j, self.R[i][j])
            for i in range(self.numUsers)
            for j in range(self.numItems)
            if self.R[i][j] > 0
        ]

        training_process = []
        for i in range(self.iterations):
            random.shuffle(self.samples)
            self.sgd()
            # mse = self.mse()
            # training_process.append((i, mse))
            # if (i+1) % 10 == 0:
            #     print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        xs, ys = self.np.nonZeroIndexes(self.R)
        predicted = self.fullMat()
        error = 0

        for x, y in zip(xs, ys):
            error += pow(self.R[x][y] - predicted[x][y], 2)
        return math.sqrt(error)

    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.bu[i] += self.alpha * (e - self.beta * self.bu[i])
            self.bi[j] += self.alpha * (e - self.beta * self.bi[j])

            self.P[i] = self.np.add(self.P[i], self.np.multiply(
                self.np.substract(self.np.multiply(self.Q[j], 2 * e), self.np.multiply(self.P[i], self.beta)), self.alpha))
            self.Q[j] = self.np.add(self.Q[j], self.np.multiply(
                self.np.substract(self.np.multiply(self.P[i], 2 * e), self.np.multiply(self.Q[j], self.beta)), self.alpha))

    def get_rating(self, i, j):
        prediction = self.b + self.bu[i] + self.bi[j] + self.np.dot(self.P[i], self.Q[j])
        return prediction

    def fullMat(self):
        return [[self.get_rating(i, j) for j in range(self.numItems)] for i in range(self.numUsers)]


io = Io()
R = io.input()

mf = MF(R, K, alpha, beta, numIter)
training_process = mf.train()

Rd, Rk = io.sortResults(R, mf.fullMat())
io.printDict(Rd, Rk)
