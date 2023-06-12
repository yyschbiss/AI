import pandas
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn import metrics
import joblib

import numpy as np


traindata = np.loadtxt("E:\MDCFA论文\RF\\traindata\\train.txt", usecols=np.arange(0,15))
trainlabel = np.loadtxt("E:\MDCFA论文\RF\\traindata\\train.txt", usecols=(15,))
# print(traindata.shape)
# print(trainlabel.shape)


testdata = np.loadtxt("E:\MDCFA论文\RF\\testdata\GG1-1-3.txt", usecols=np.arange(0,15))
testlabel = np.loadtxt("E:\MDCFA论文\RF\\testdata\GG1-1-3.txt", usecols=(15,))
# testXYZ = np.loadtxt("F:/temp/DATA/feature/QZ4-1-feature-0.1.txt",usecols=np.arange(1,4))
# testIntensity = np.loadtxt("F:/temp/DATA/feature/QZ4-1-feature-0.1.txt",usecols=(8,))

print(testdata.shape)
print(testlabel.shape)
# print(testXYZ.shape)
# print(testIntensity.shape)
#X = pandas.read_csv("C:/Users/pc/Desktop/PY/QZ4-Feature.txt")
#y = pandas.read_csv("C:/Users/pc/Desktop/PY/QZ4-Label.txt")

#digits = load_digits()
#dir(digits)
#df = pandas.DataFrame(digits.data)
#df.head()
#df['target'] = digits.target
#X = df.drop('target',axis='columns')
#y = df.target

# data = np.loadtxt("E:/DATA/demo/OptimalRadius/GG1-1-feature-optimal.txt",usecols=np.arange(8,18))
# label = np.loadtxt("E:/DATA/demo/OptimalRadius/GG1-1-feature-optimal.txt",usecols=(7,))
# ID = np.loadtxt("E:/DATA/demo/OptimalRadius/GG1-1-feature-optimal.txt",usecols=(0,))
# traindata, testdata, trainlabel, testlabel,other,testID = train_test_split(data,label,ID,test_size=0.9) #spliting the data into 2 parts



forestClassifer = RandomForestClassifier(n_estimators=100) #random forest
forestClassifer.fit(traindata, trainlabel)
joblib.dump(forestClassifer, 'E:\MDCFA论文\RF\\traindata\\train.pkl')

# forestClassifer = joblib.load('E:\MDCFA论文\RF\\traindata\\train.pkl')
forestClassifer.score(testdata, testlabel)
y_predictedForest = forestClassifer.predict(testdata)

print("this is the predicted y with the random forest method \n")
print(y_predictedForest,"\n")
print(y_predictedForest.shape)
confusionMatrixRandomForest = confusion_matrix(testlabel, y_predictedForest)
print("this is the confusion matrix with the random forest method \n ")
print(confusionMatrixRandomForest,"\n")


# numpy.savetxt(test.txt,result)
# classifierGradientBoosting = GradientBoostingClassifier(n_estimators=20) #gradient boosting
# classifierGradientBoosting.fit(traindata, trainlabel)
# classifierGradientBoosting.score(testdata, testlabel)
# y_predictedGradient = classifierGradientBoosting.predict(testdata)
# print("this is the predicted y with the gradient boosting method ")
# print(y_predictedGradient,"\n")
# confusionMatrixGradientBoosting = confusion_matrix(testlabel, y_predictedGradient)
# print("this is the confusion matrix with the gradient boosting method\n")
# print(confusionMatrixGradientBoosting,"\n")


print("this is the recall score for the random forest classification") #random forest scores
randomForestRecallScore = recall_score(testlabel, y_predictedForest,average = 'micro')
print(randomForestRecallScore)
print("F1 score for the random forest ")
f1ScoreRandomForest = metrics.f1_score(testlabel, y_predictedForest,average='micro')
print(f1ScoreRandomForest)
print("accuracy score for random forest ")
accuracyScoreRandomForest = metrics.accuracy_score(testlabel, y_predictedForest)
print(accuracyScoreRandomForest,"\n")

np.savetxt('result.txt', y_predictedForest, fmt='%d',delimiter='\t')

# resultXYZ = np.array([], dtype = float)
# resultIntensity = np.array([], dtype = float)
# for (i, value) in enumerate(y_predictedForest):
#     if value == 1:
#         resultXYZ = np.append(resultXYZ,testXYZ[i])
#         resultIntensity = np.append(resultIntensity,testIntensity[i])
# print(resultXYZ.shape)
# print(resultXYZ,"\n")
# print(resultIntensity.shape)
# print(resultIntensity,"\n")
# np.savetxt('test.txt',resultXYZ,fmt='%f')






# print("this is the recall score for the gradient boosting classification") #gradient boosting score
# gradientBoostingRecallScore = recall_score(testlabel, y_predictedGradient,average = 'micro')
# print(gradientBoostingRecallScore)
# print("F1 score for the gradient boosting")
# f1ScoreGradientBoosting = metrics.f1_score(testlabel, y_predictedGradient,average='micro')
# print(f1ScoreGradientBoosting)
# print("accuracy score for gradient boosting")
# accuracyScoreGradientBoosting = metrics.accuracy_score(testlabel, y_predictedGradient)
# print(accuracyScoreGradientBoosting)