import numpy as np
from sklearn.metrics import confusion_matrix,cohen_kappa_score

data = np.loadtxt('E:\MDCFA论文\论文\修改\可移植性\数据3\wood_leaf_小地块\第二次特征\\SMaple_S1.txt')
label = data[:,3:4]
leaf = np.where(label == 0)[0]
stem = np.where(label == 1)[0]
print(len(leaf) + len(stem), len(stem), len(leaf), len(stem) / len(leaf))


# data = np.loadtxt('E:\MDCFA论文\论文\修改\可移植性\数据3\wood_leaf_小地块\\noground\\TAspen_S12.txt')
# xyzi = data[:, :4]
# label = data[:, 4:5]
#
# c1 = np.where(label == 1)[0]
# c2 = np.where(label == 2)[0]
# c3 = np.where(label == 3)[0]
#
# xyzi1 = xyzi[c1, :]
# xyzi2 = xyzi[c2, :]
# xyzi3 = xyzi[c3, :]
#
# label1 = np.ones((len(c1), 1))
# label2 = np.ones((len(c2), 1))
# label3 = np.zeros((len(c3), 1))
#
# data1 = np.concatenate((xyzi1,label1),-1)
# data2 = np.concatenate((xyzi2,label2),-1)
# data3 = np.concatenate((xyzi3,label3),-1)
#
# result = np.concatenate((data1, data2, data3), 0)
# np.savetxt('E:\MDCFA论文\论文\修改\可移植性\数据3\wood_leaf_小地块\第二次标签\\TAspen_S12.txt', result, fmt='%f', delimiter='\t')

# data = np.loadtxt('E:\MDCFA论文\论文\修改\新建文件夹\\c.txt')
# pre = data[:, 3]
# truth = data[:, 4]
#
# confusions = confusion_matrix(truth, pre)
# TP = np.diagonal(confusions, axis1=-2, axis2=-1)
# TP_plus_FN = np.sum(confusions, axis=-1)
# TP_plus_FP = np.sum(confusions, axis=-2)
#
# acc = np.sum(pre == truth) / len(truth)
# IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)
# mIoU = np.mean(IoU)
# precision = TP / TP_plus_FP
# recall = TP / TP_plus_FN
#
# print(TP,acc,mIoU,IoU,precision,recall)
# print(cohen_kappa_score(truth, pre))



# data = np.loadtxt('E:\MDCFA论文\论文\修改\可移植性\数据3\wood_leaf_小地块\第二次特征\总体强度归一化\\111.txt')
# xyz = data[:,:3]
# label = data[:,3:4]
# intensity = data[:,4:5]
# sodl = data[:,5:6]
#
# mean = np.mean(intensity)
# stand = np.std(intensity)
# intensity = (intensity - mean) / stand
#
# result = np.concatenate((xyz, label, intensity, sodl), -1)
# np.savetxt('E:\MDCFA论文\论文\修改\可移植性\数据3\wood_leaf_小地块\第二次特征\总体强度归一化\\result.txt', result, fmt='%f', delimiter='\t')

# data = np.loadtxt('E:\MDCFA论文\论文\修改\可移植性\数据3\wood_leaf_小地块\第二次特征\\LPine_S1.txt')
# xyz = data[:,:3]
# label = data[:,3:4]
# intensity = np.ones((len(label), 1))
# sodl = data[:,5:6]
#
# result = np.concatenate((xyz, intensity, sodl), -1)
#
# choice_1 = np.where(label == 0)[0]
# choice_2 = np.where(label == 1)[0]
#
# leaf = result[choice_1,:]
# stem = result[choice_2,:]
#
# np.savetxt('E:\MDCFA论文\论文\修改\可移植性\数据3\wood_leaf_小地块\第二次特征\相同强度值_data3-4\\LPine_S1-leaf.txt', leaf, fmt='%f', delimiter='\t')
# np.savetxt('E:\MDCFA论文\论文\修改\可移植性\数据3\wood_leaf_小地块\第二次特征\相同强度值_data3-4\\LPine_S1-stem.txt', stem, fmt='%f', delimiter='\t')



