#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from numpy import *
import re
#import csv

#读取短信数据
def readTxt():
    #arraylines = csv.reader(open('D:/研一课程/模式识别/lintcode数据/messages/train.csv', encoding='UTF-8'))
    text=open(r'D:\\message.txt',encoding='gb18030',errors='ignore')
    arraylines=text.readlines()
    docList=[]
    classList=[]
    for line in arraylines:
       docList.append(textParse(line[:-3]))
       try:
            classList.append(int(line[-2]))
       except:
           classList.append(1)
    return docList,classList

#文档分词函数
#将单词长度小于等于2的过滤掉，并且将其变成小写字母。
def textParse(bigString):
    listOfTokens = re.split(r'\W*' , bigString)  # re.split，支持正则及多个字符切割
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


#创建一个带有所有单词的词汇列表
def createVocabList(dataSet):
      vocabSet = set([])
      for document in dataSet:
          vocabSet = vocabSet | set(document)
      return list(vocabSet)

#创建一个只含1或0的邮件向量
def Wordtovector(vocablist,inputset):
    returnvec=[0]*len(vocablist)
    for word in inputset:
        returnvec[vocablist.index(word)]+=1
    return returnvec

def setOfWords2Vec(vocabList, inputSet):
      retVocabList = [0] * len(vocabList)
      for word in inputSet:
          if word in vocabList:
              retVocabList[vocabList.index(word)] += 1
          else:
              print('word ',word ,'not in dict')
      return retVocabList

#文档词袋模型
#vocablist为词汇表，inputSet为输入的邮件
def bagOfWords2VecMN(vocabList, inputSet):
      returnVec = [0]*len(vocabList)#大小与词向量相同
      for word in inputSet:
          if word in vocabList:
              returnVec[vocabList.index(word)] += 1#查找单词的索引
      return returnVec

#训练函数
#trainMat是训练样本的词向量，可以看做一个矩阵，他的每一行为一个短信的词向量
#trainGategory为与trainMat对应的类别，值为0表示正常，1表示垃圾短信
def trainNB0(trainMatrix,trainCatergory):
      numTrainDoc = len(trainMatrix)
      numWords = len(trainMatrix[0])#词汇表长度
      pAbusive = sum(trainCatergory)/float(numTrainDoc)
      #防止多个概率的成绩当中的一个为0
      p0Num = ones(numWords)
      p1Num = ones(numWords)
      p0Denom = 2.0
      p1Denom = 2.0
      for i in range(numTrainDoc):
          if trainCatergory[i] == 1:
              p1Num +=trainMatrix[i]#统计垃圾短信类中每个单词的个数
              p1Denom += sum(trainMatrix[i])#计算垃圾短信中的单词总数
          else:
              p0Num +=trainMatrix[i]
              p0Denom += sum(trainMatrix[i])
      p1Vect = log(p1Num/p1Denom)#计算垃圾短信中每个单词概率
                                 #出于精度的考虑，否则很可能到限归零
      p0Vect = log(p0Num/p0Denom)
      return p0Vect, p1Vect, pAbusive

#分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
      p1 = sum(vec2Classify * p1Vec) + log(pClass1)
      p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
      if p1 > p0:
          return 1
      else:
          return 0

#测试函数
def spamTest():
      docList, classList = readTxt()
      vocabList = createVocabList(docList)
      #lendoc = len(docList)
      trainingSet = range(5572)
      testSet = []
      for i in range(1000):
          randindex = int(random.uniform(0,5572))
          testSet.append(trainingSet[randindex])
      trainMat = []
      for j in range(5572):
        #trainMat.append(Wordtovector(vocabList, docList[j]))
        #trainMat.append(setOfWords2Vec(vocabList, docList[j]))
        trainMat.append(bagOfWords2VecMN(vocabList, docList[j]))

      p0V,p1V,pSpam = trainNB0(array(trainMat),array(classList))
      errorCount = 0
      for docIndex in testSet:        #classify the remaining items
          wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
          #wordVector = setOfWords2Vec(vocabList, docList[docIndex])
          #wordVector = Wordtovector(vocabList, docList[i])
          if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
              errorCount += 1
             # print("classification error",docList[i])
      print('the error rate is: ',float(errorCount)/len(testSet))
      #return vocabList,fullText


def main():
     spamTest()

if __name__ == '__main__':
     main()