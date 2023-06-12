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

#读取训练数据和标签
traindata = np.loadtxt("E:\MDCFA论文\RF\\traindata\\train.txt", usecols=np.arange(3,15))
trainlabel = np.loadtxt("E:\MDCFA论文\RF\\traindata\\train.txt", usecols=(15,))


#读取测试数据和标签
testdata = np.loadtxt("E:\MDCFA论文\RF\\testdata\LB3-1-3.txt", usecols=np.arange(3,15))
testlabel = np.loadtxt("E:\MDCFA论文\RF\\testdata\LB3-1-3.txt", usecols=(15,))
#读取测试数据xyz坐标，用于最后可视化输出
testXYZ = np.loadtxt("E:\MDCFA论文\RF\\testdata\LB3-1-3.txt", usecols=np.arange(0,3))


print(testdata.shape)
print(testlabel.shape)
print(testXYZ.shape)


#只有在第一次训练模型时需要使用下面三句代码，模型会存储到对应路径，下次使用时可直接加载模型。
forestClassifer = RandomForestClassifier(n_estimators=100) #random forest
forestClassifer.fit(traindata, trainlabel)
joblib.dump(forestClassifer, 'E:\MDCFA论文\RF\\traindata\\train.pkl')


#加载模型，第一次训练模型需要注释掉这一步
forestClassifer = joblib.load('E:\MDCFA论文\RF\\traindata\\train.pkl')
#将测试数据放入模型
forestClassifer.score(testdata, testlabel)
#结果预测
y_predictedForest = forestClassifer.predict(testdata)


#精度评价：

total_seen_class = [0 for _ in range(2)]
total_correct_class = [0 for _ in range(2)]
total_pred_class = [0 for _ in range(2)]
total_iou_deno_class = [0 for _ in range(2)]
for l in range(2):
    total_seen_class[l] += np.sum((testlabel == l))
    total_pred_class[l] += np.sum((y_predictedForest == l))
    total_correct_class[l] += np.sum((y_predictedForest == l) & (testlabel == l))
    total_iou_deno_class[l] += np.sum(((y_predictedForest == l) | (testlabel == l)))

correct = np.sum((y_predictedForest == testlabel))
mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))

OA = correct / len(testlabel)
print('OA: %f' % (OA))
print('mIoU: %f' % (mIoU))
print('wood precision: %f' % (total_correct_class[1] / total_pred_class[1]))
print('wood recall: %f' % (total_correct_class[1] / total_seen_class[1]))
print('leaf precision: %f' % (total_correct_class[0] / total_pred_class[0]))
print('leaf recall: %f' % (total_correct_class[0] / total_seen_class[0]))

#结果输出
label_matrix = np.matrix(y_predictedForest)
result_label = np.transpose(label_matrix)
testlabel = testlabel.reshape((-1, 1))
#x,y,z,预测标签，真实标签
result_XYZL = np.concatenate((testXYZ, result_label, testlabel), axis=1)
np.savetxt('E:\MDCFA论文\RF\\result\\LB3-1-3.txt', result_XYZL, fmt='%f',delimiter='\t')








