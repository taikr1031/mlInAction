from numpy import *
from os import listdir
import operator


########################## 【k-近邻算法】测试数据集 ##########################
def createDataSet():
    # group = array([[1, 0], [1, 0], [0, 1], [0, 0]])
    group = array([[1.0, 1.1], [1.0, 1.0], [0., 0.], [0., 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDataSet()


########################## 【k-近邻算法】 ##########################
def classify0(inX, dataSet, labels, k):  # inX:输入向量;dataSet:训练样本集;labels:标签向量;k:最近邻居数目
    dataSetSize = dataSet.shape[0];  # 矩阵第一维度的维度(行数)(0 按列计算；1 按行计算)
    # tile函数将inX=[0,0]第一个维度(列)重复1遍，第二个维度(行)重复dataSetSize遍，然后与dataSet矩阵相减
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2  # 矩阵按元素逐个平方，将所有负数变成正数(负负得正)
    sqDistances = sqDiffMat.sum(axis=1)  # 矩阵的每一行元素相加(0 按列计算；1 按行计算)
    distances = sqDistances ** 0.5  # 矩阵按元素逐个开方，2个点之间正数的距离
    sortedDistIndicies = distances.argsort()  # =[2 3 1 0] 矩阵值从小到大的索引值，即原始矩阵的排序索引值
    classCount = {}  # 声明字典类型
    for i in range(k):  # 循环从0到k(不包含k)  选择距离最小的k个点
        voteIlabel = labels[sortedDistIndicies[i]]  # =B B A
        print("voteIlabel %d: %s" % (i, voteIlabel))
        # ={'A': 1, 'B': 2} 如voteIlabel重复，则classCount元祖中key为voteIlabel的值加1，如'B':2是因为循环voteIlabel中有2个B
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    '''
    =[('B', 2), ('A', 1)]
    classCount.items(): 将classCount字典分解成元祖列表
    key = operator.itemgetter(1): 按第二个元素的次序对元祖排序
    reverse=True： 降序，从大到小，返回发生频率最高的元素标签
    '''
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(classCount)
    print(sortedClassCount)
    return sortedClassCount[0][0]  # 取数组中第一行中第一列的元素B


resultA = classify0([0.9, 1.2], group, labels, 3)
resultB = classify0([0, 0], group, labels, 3)
print("【k-近邻算法】")
print(resultA)
print(resultB)


########################## 【约会网站】将文本文件转换成矩阵 ##########################
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # 文本文件行数
    returnMat = zeros((numberOfLines, 3))  # 创建以零填充元素值=0，元素个数=3，维度=numberOfLines的二维矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 删除字符串头尾的空格和回车字符
        listFromLine = line.split('\t')  # 将整行数据分割成元素列表
        returnMat[index, :] = listFromLine[0: 3]  # 取元素列表的前3个元素，填充到特征矩阵returnMat的第index行中
        classLabelVector.append(int(listFromLine[-1]))  # 索引值-1表示列表中最后一列元素
        index += 1
    return returnMat, classLabelVector


########################## 【约会网站】根据矩阵形成散点图 ##########################
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat, datingLabels = file2matrix('./doc/datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels)
# 0：每年飞行里程数；1：玩游戏所耗时间百分比；3：每周冰淇淋消耗公升数
# 无样本类别标签（单色，大小相同）
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
# 有样本类别标签（多色，大小不同）
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))  # 1    2

ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))  # 0  1
plt.show()


########################## 【约会网站】归一化特征值，自动将数字特征值转化为0到1的区间 ##########################
# 归一化公式： newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 第0列最小值，不是当前行的
    maxVals = dataSet.max(0)  # 第0列最大值，不是当前行的
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


print("【约会网站】转化为归一化特征值")
normMat, ranges, minVals = autoNorm(datingDataMat)
print(normMat)


########################## 【约会网站】测试 ##########################
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('./doc/datingTestSet2.txt')  # 将文本文件转换成矩阵
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 转化为归一化特征值
    m = normMat.shape[0]  # 矩阵第一维度的维度(行数)(0 按列计算；1 按行计算)
    numTestVecs = int(m * hoRatio)  # 测试向量数量(10%)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


print("【约会网站】测试")
datingClassTest()


########################## 【约会网站】预测函数 ##########################
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spend playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("./doc/datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


print("【约会网站】预测函数")
classifyPerson()


########################## 【手写数字识别】图像转化为测试向量 ##########################
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


print("【手写数字识别】图像转化为测试向量")
testVector = img2vector("./doc/digits/testDigits/0_13.txt")
print(testVector[0, 1:31])


########################## 【手写数字识别】测试 ##########################
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("./doc/digits/trainingDigits")  # 获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("./doc/digits/trainingDigits/%s" % fileNameStr)
    testFileList = listdir("./doc/digits/testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector("./doc/digits/testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
        print("\nthe total number of errors is: %d" % errorCount)
        print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


handwritingClassTest()
