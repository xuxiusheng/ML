import numpy as np
import random
import matplotlib.pyplot as plt

def load_data(filename):
    dataMat = []
    yMat = []
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataMat.append([float(line[0]), float(line[1])])
        yMat.append(float(line[2]))
    return np.array(dataMat), np.array(yMat)

def smoSimple(dataArr, yArr, C, toler, maxIter):
    numSample, numDim = dataArr.shape
    b = 0
    alphas = np.zeros((numSample, 1))
    iterations = 0

    while iterations < maxIter:

        alphaPairsChanged = 0
        for i in range(numSample):
            fxi = np.sum(alphas * yArr[:, np.newaxis] * dataArr * dataArr[i, :]) + b
            Ei = fxi - yArr[i]

            if ((yArr[i] * Ei < -toler) and (alphas[i] < C)) or ((yArr[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, numSample)
                fxj = np.sum(alphas * yArr[:, np.newaxis] * dataArr * dataArr[j, :]) + b
                Ej = fxj - yArr[j]
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                if yArr[i] != yArr[j]:
                    L = max(0, alphaJold - alphaIold)
                    H = min(C, C + alphaJold - alphaIold)
                else:
                    L = max(0, alphaIold + alphaJold - C)
                    H = min(C, alphaIold + alphaJold)
                if L == H:
                    print("L == H")
                    continue
                eta = dataArr[i, :].dot(dataArr[i, :]) + dataArr[j, :].dot(dataArr[j, :]) - 2 * \
                      dataArr[i, :].dot(dataArr[j, :])
                if eta <= 0:
                    print("eta <= 0")
                    continue
                alphas[j] = alphaJold + yArr[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j is not moving enough")
                    continue
                alphas[i] = alphaIold + yArr[i] * yArr[j] * (alphaJold - alphas[j])

                bi = b - Ei - yArr[i] * dataArr[i, :].dot(dataArr[i, :]) * (alphas[i] - alphaIold) - yArr[j] * \
                     dataArr[i, :].dot(dataArr[j, :]) * (alphas[j] - alphaJold)

                bj = b - Ej - yArr[i] * dataArr[i, :].dot(dataArr[j, :]) * (alphas[i] - alphaIold) - yArr[j] * \
                     dataArr[j, :].dot(dataArr[j, :]) * (alphas[j] - alphaJold)

                if 0 < alphas[i] < C:
                    b = bi
                elif 0 < alphas[j] < C:
                    b = bj
                else:
                    b = (bi + bj) / 2
                alphaPairsChanged = alphaPairsChanged + 1
        if alphaPairsChanged == 0:
            iterations +=1
        else:
            iterations = 0
    print(alphas)
    print(b)
    return b, alphas


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif L > aj:
        aj = L
    else:
        aj = aj
    return aj


def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j



def testSVM():
    dataArr, yArr = load_data('testSet.txt')
    C = 0.6
    toler = 0.001
    maxIter = 40
    b, alphas = smoSimple(dataArr, yArr, C, toler, maxIter)
    return b, alphas

def showData(filename, line=None):
    dataArr, yArr = load_data(filename)
    data_class_1_index = np.where(yArr == -1)
    data_class_1 = dataArr[data_class_1_index, :].reshape(-1, 2)

    data_class_2_index = np.where(yArr == 1)
    data_class_2 = dataArr[data_class_2_index, :].reshape(-1, 2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data_class_1[:, 0], data_class_1[:, 1], c='r', label="$-1$")
    ax.scatter(data_class_2[:, 0], data_class_2[:, 1], c='g', label="$+1$")
    plt.legend()
    if line is not None:
        b, alphas = line
        x = np.linspace(1, 8, 50)
        w = np.sum(alphas * yArr[:, np.newaxis] * dataArr, axis=0)
        # print(w.shape)
        y = np.array([(-b - w[0] * x[i]) / w[1] for i in range(50)])
        y1 = np.array([(1 - b - w[0] * x[i]) / w[1] for i in range(50)])
        y2 = np.array([(-1 - b - w[0] * x[i]) / w[1] for i in range(50)])
        ax.plot(x, y, 'b-')
        ax.plot(x, y1, 'b--')
        ax.plot(x, y2, 'b--')

    plt.show()



if __name__ == '__main__':
    filename = 'testSet.txt'
    b, alphas = testSVM()
    showData(filename, (b, alphas))