from numpy import *
import feedparser


########################## 【训练程序自动检测侮辱性留言的训练数据集字典】 ##########################
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 文档标签，人工标注。0:侮辱言论，1:正常言论。其中1、3、5行是正常言论；2、4、6行是侮辱言论
    return postingList, classVec


########################## 【创建一个包含在所有文档中出现的不重复词的列表】 ##########################
# @param    : dataSet   训练样本集
# @return   : vocabSet  将训练样本集dataSet多维列表中的所有元素去重后合并成一个shape=(32,)(一维32列)的列表 ['my', 'dog' ... 'food', 'stupid']
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 创建2个集合的并集
    return list(vocabSet)


########################## 【循环inputSet中的每个单词，如出现在vocabList中，将该单词的位置的元素值[词集：变为1/词袋：加1]，默认全部都是0】 ##########################
# 词集模型(set-of-words-model)：每个单词出现与否 [1, 0, 1 ... 0, 0]
# @param vocabList  : createVocabList()->@return: vocabSet
# @param inputSet   : 待分析的文档。即循环读取训练样本集中的一行列表
# @return returnVec : shape=(32,)列表，每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现 [1, 0, 1 ... 0, 0]
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都是0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 词袋模型(bag-of-words-model)：每个单词出现次数 [2, 0, 3 ... 1, 0]
# @param vocabList  : createVocabList()->@return: vocabSet
# @param inputSet   : 待分析的文档。即循环读取训练样本集中的一行列表
# @return returnVec : shape=(32,)列表，每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现 [1, 0, 1 ... 0, 0]
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都是0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# print("【从文本中构建词向量】")
# listOposts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOposts)
# print(myVocabList)
# print(setOfWords2Vec(myVocabList, listOposts[0]))
# print(setOfWords2Vec(myVocabList, listOposts[3]))
# print()


########################## 【朴素贝叶斯分类器训练函数-计算分类所需概率】 ##########################
# @param trainMatrix  : shape=(32,6)列表，setOfWords2Vec()->@return returnVec
# @param trainCategory   : loadDataSet()->classVec = [0, 1, 0, 1, 0, 1]
# @return p0Vect : shape=(32,1)列表，[-3.04452244 -2.35137526 ... -3.04452244]
# @return p1Vect : shape=(32,1)列表，[-2.56494936 -3.25809654 ... -2.56494936]
# @return pABusive :=0.5 侮辱类概率
def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # =6 矩阵行数
    numWords = len(trainMatrix[0])  # =32 矩阵列数，第一行中单词个数
    pABusive = sum(trainCategory) / float(numTrainDocs)  # 3/6=0.5
    p0Num = ones(numWords)  # 初始化概率，分子
    p1Num = ones(numWords)  # 初始化概率，分子
    p0Denom = 2.0  # 初始化概率，分母
    p1Denom = 2.0  # 初始化概率，分母
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 正常言论 trainCategory = [0, 1, 0, 1, 0, 1]
            p1Num += trainMatrix[i]  # 2个列表中对应顺序的元素值相加，得到该文档中所有单词对应在postingList的42个词汇表中出现的次数列表
            p1Denom += sum(trainMatrix[i])  # =19 正常词汇出现的总次数
        else:  # 侮辱言论
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])  # =7
    p1Vect = log(p1Num / p1Denom)  # [ 3.  1.  1. ... 2.  1.] / 19
    p0Vect = log(p0Num / p0Denom)  # 上行的列表中的每个元素除以出现总次数，得到条件概率
    return p0Vect, p1Vect, pABusive


# print("【朴素贝叶斯分类器训练函数】")
# listOposts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOposts)
# trainMat = []
# for postinDoc in listOposts:
#     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# p0V, p1V, pAb = trainNBO(trainMat, listClasses)
# print("文档属于侮辱类概率 pAb：%f" % pAb)
# print(p0V)
# print(p1V)
# print()


########################## 【朴素贝叶斯分类函数 1】 ##########################
# @param vec2Classify  : shape=(32,1)列表，待分类列表
# @param p0Vec   : trainNBO()->p0Vect
# @param p1Vec   : trainNBO()->p1Vect
# @param pClass1   : trainNBO()->pABusive = 0.5
# @return:  根据计算出来的2个概率，取最大值返回是否是侮辱言论
# 将要判断的单词(单词都属于词汇表的一部分)列表放入训练函数结果中的2个类别概率列表中分别比较，看这些单词在哪个类别中出现的概率更大（出现的次数最多）
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # vec2Classify列表中1的数量和位置，由待分类列表中的单词在词汇表中是否存在以及位置决定。
    # sum(vec2Classify * p1Vec)=sum([0 1 ... 1] * [[-2.56494936 -3.25809654 ... -2.56494936]])
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


########################## 【朴素贝叶斯分类函数 2】 ##########################
def testingNB():
    listOposts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOposts)  # 创建一个包含在所有文档中出现的不重复词的列表
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNBO(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']  # 只要测试列表中哪怕其中只有一个侮辱性词汇，则整个测试列表都认为是侮辱性的
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))


# print("【朴素贝叶斯分类函数】")
# testingNB()
# print()


def textParse(bigString):
    import re
    # print(type(bigString))
    if isinstance(bigString, bytes):
        bigString = bigString.decode("GBK")
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


########################## 【文本解析及完整的垃圾邮件测试函数】 ##########################
def spamTest():
    docList = []
    classList = []  # 用来与算法结果比对的已知测试检验结果 shape(0,50) [1,0,1,0, ... 1,0,1,0,]
    fullText = []
    for i in range(1, 26):  # 导入并解析文本文件
        wordList = textParse(open("./doc/email/spam/%d.txt" % i, 'rb').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open("./doc/email/ham/%d.txt" % i, 'rb').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # docList.shape(50,n); vocabList.shape(0,b*50=694)
    trainingSet = range(50)
    testSet = []
    # 这种随机选择数据的一部分作为训练集，而剩余部分作为测试集的过程叫做留存交叉验证(hold-out cross validation)
    for i in range(10):  # 从50个训练样本集中随机选择10个作为测试样本集，同时从训练集中剔除
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (list(trainingSet)[randIndex])  # 将测试样本集从50个训练样本集中删除
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # 训练样本集 len(trainingSet)=50
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNBO(array(trainMat),
                               array(trainClasses))  # 概率计算 trainMat.shape(50,694); trainClasses.shape(1,50)
    errorCount = 0
    for docIndex in testSet:  # 循环测试样本集，对每封邮件进行分类 testSet.shape(1, 10) [11, 39, 10, 28, 24, 3, 29, 36, 33, 13]
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is: ", float(errorCount / len(testSet)))


# print("【文本解析及完整的垃圾邮件测试函数】")
# spamTest()
# spamTest()
# print()


########################## 【RSS源分类器及高频词去除函数 1】 ##########################
def calcMostFreq(vocabList, fullText):  # 计算词汇表中每个单词在文档中出现次数，并提取频率最高的前30个单词
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]  # len=30 {'also': 1, 'life': 2 ... 'etc': 2}


########################## 【RSS源分类器及高频词去除函数 2】 ##########################
def localWords(feed1, feed0):
    # import feedparser
    docList = [];
    classList = [];
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])  # 每次访问一条RSS源
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)  # shape(50,n)
        fullText.extend(wordList)  # shape(1, 50*n=1623)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])  # 去掉出现次数最高的那些词
    trainingSet = range(2 * minLen);
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (list(trainingSet)[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNBO(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


# print("【RSS源分类器及高频词去除函数】")
# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# vocabList, pSF, pNY = localWords(ny, sf)
# print(vocabList)
# vocabList, pSF, pNY = localWords(ny, sf)
# print(vocabList)
# print(pSF)
# print(pNY)
# print()


########################## 【最具表征性的词汇显示函数】 ##########################
def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


# print("【最具表征性的词汇显示函数】")
# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# getTopWords(ny, sf)
# print()
