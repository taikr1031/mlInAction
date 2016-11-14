from math import log  # log=对数。比如log（2）8=3，3叫做以2为底8的对数
# from treePlotter import *
import operator
import treePlotter


########################## 【鱼鉴定数据集】 ##########################
def createDataSet():  # ['不浮出水面是否可以生存', '是否有脚蹼', '属于鱼类']
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']  # ['不浮出水面是否可以生存', '是否有脚蹼']
    return dataSet, labels


########################## 【计算数据集的香农熵】 ##########################
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}  # 所有可能的分类的字典
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 该字典的键是dataSet的最后一列的值
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算每个类标签的出现概率
        shannonEnt -= prob * log(prob, 2)  # 用上方的概率计算香农熵，以2为底求对数，每次循环用shannonEnt与结果相减.log对数为负数
    return shannonEnt


print("【计算数据集的香农熵】")
myDat, labels = createDataSet()
print(myDat)
result = calcShannonEnt(myDat)
print(result)
myDat[0][-1] = 'maybe'
print(myDat)
result = calcShannonEnt(myDat)
print(result)


########################## 【按给定特征划分数据集】 ##########################
# dataSet: 待划分的数据集
# axis: 划分数据集的特征
# value: 特征的返回值
def splitDataSet(dataSet, axis, value):
    retDataSet = []  # python传递的是列表的引用，在函数内部对列表对象的修改，会影响列表对象的整个生命周期。为了不修改原始数据集，创建一个新列表对象
    for featVec in dataSet:
        if featVec[axis] == value:  # 用featVec的第axis列元素的值和value比较是否相等
            reducedFeatVec = featVec[:axis]  # = [] 取featVec列表中从开始列到第axis(0)列的元素
            reducedFeatVec.extend(featVec[axis + 1:])  # 取featVec列表中从第axis + 1(1)列到结束列的元素
            retDataSet.append(reducedFeatVec)
    return retDataSet


print("【按给定特征划分数据集】")
print(splitDataSet(myDat, 0, 1))  # =[[1, 'yes'], [1, 'yes'], [0, 'no']]


########################## 【通过计算信息熵计算最好的数据集划分特征对应列数】 ##########################
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # =2 dataSet[0]=[1,1,'yes'] dataSet第一行数据
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 循环dataSet的每一列
        featList = [example[i] for example in dataSet]  # =[1,1,0,1,1,]取dataSet列表中每行第i列的所有元素列表
        uniqueVals = set(featList)  # = {0,1} 去重，创建唯一的分类标签列表
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算每种划分方式的信息熵
        infoGgain = baseEntropy - newEntropy
        if (infoGgain > bestInfoGain):
            bestInfoGain = infoGgain
            bestFeature = i
    return bestFeature


print("【通过计算信息熵计算最好的数据集划分特征对应列数】")
myDat, labels = createDataSet()
print(chooseBestFeatureToSplit(myDat))
print(myDat)


########################## 【获取列表中出现次数最多的类别值】 ##########################
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


########################## 【创建树函数】 ##########################
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # =['yes', 'yes', 'no', 'no', 'no'] 取dataSet列表中每行最后一列的所有元素列表
    if classList.count(classList[0]) == len(classList):  # count() 方法用于统计某个元素在列表中出现的次数
        return classList[0]  # 类别完全相同，则停止继续划分
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 遍历完所有特征时返回出现次数最多的
    bestFeat = chooseBestFeatureToSplit(dataSet)  # =0 通过计算信息熵计算最好的数据集划分特征对应列数
    bestFeatLabel = labels[bestFeat]  # ='no surfacing'
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # 特征列表中删除当前最好的元素bestFeatLabel
    featValues = [example[bestFeat] for example in dataSet]  # 取dataSet列表中每行最好划分特征列的所有元素列表
    uniqueVals = set(featValues)  # 去重
    for value in uniqueVals:  # 遍历当前选择特征包含的所有属性值
        subLabels = labels[:]  # 复制
        # splitDataSet函数返回符合按给定特征划分的数据集；subLabels是删除已递归划分过的特征值列表
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree  # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}


print("【创建树函数】")
myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print(myTree)


########################## 【使用决策树的分类函数】 ##########################
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]  # ='no surfacing'
    secondDict = inputTree[firstStr]  # ={0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    featIndex = featLabels.index(firstStr)  # =0 查找当前列表中第一个匹配firstStr变量的元素
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)  # 递归
            else:
                classLabel = secondDict[key]
    return classLabel


print("【使用决策树的分类函数】")
myDat, labels = createDataSet()
print(labels)
myTree = treePlotter.retrieveTree(0)
print(myTree)
print(classify(myTree, labels, [1, 0]))
print(classify(myTree, labels, [1, 1]))


########################## 【决策树的存储】 ##########################
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


print("【决策树的存储】")
myTree = treePlotter.retrieveTree(0)
storeTree(myTree, 'test.txt')
print(grabTree('test.txt'))


########################## 【决策树预测隐形眼镜类型】 ##########################
def printInfo():
    fr = open('./doc/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    treePlotter.createPlot(lensesTree)
    return lensesLabels, lensesTree


print("【决策树预测隐形眼镜类型】")
myTree, labels = printInfo()
print(myTree)
print(labels)
