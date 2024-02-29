import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
# 0 1 2 3 4 5

# 从CSV文件中读取数据
class_dict = {0:'DWI non-tumor',1:'DWI tumor', 2:'T2WI non-tumor', 3:'T2WI tumor', 4:'CE-T1WI non-tumor', 5:'CE-T1WI tumor'}
model_dict = {0:'ConvNeXt-T (1k)',1:'ConvNeXt-XL (22k)', 2:'ConvNeXt-B (1k)', 3:'ConvNeXt-B (22k)', 4:'ConvNeXt-L (1k)', 5:'ConvNeXt-L (22k)', 6:'onvNeXt-S (1k)'}


# ConvNeXt-T (1k)
# ConvNeXt-S (1k)
# ConvNeXt-B (1k)
# ConvNeXt-B (22k)
# ConvNeXt-L (1k)
# ConvNeXt-L (22k)
# ConvNeXt-XL (22k)
for key, value in model_dict.items():

    model_name= value
    print(model_name)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    matrix_list = []
    label_list = []
    for i in range(6):
        filepath = os.path.join('./predict_result'+str(key), str(i)+'.csv')
        df = pd.read_csv(filepath)
        # df = pd.read_csv(filepath, skiprows=1)
        df = df.iloc[:, :-1]
        matrix = df.values
        # print(matrix.shape[0])
        label_list.append(np.full_like(matrix[:,0], i))
        matrix_list.append(matrix)

    y_scores = result_matrix = np.concatenate(matrix_list, axis=0)
    y_true = label_matrix = np.concatenate(label_list, axis=0)

    n = 2678

    for i in range(y_scores.shape[1]):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        value = roc_auc[i]

        se = np.sqrt((roc_auc[i] * (1 - roc_auc[i])) / n)

        # 计算95%置信区间的边界
        z_value = norm.ppf(0.975)  # 对应于95%的Z值
        margin_of_error = z_value * se

        # 计算95%置信区间
        confidence_interval = (value - margin_of_error, value + margin_of_error)
        # confidence_interval_high = value + margin_of_error

        # print("Test Set AUC: {:.6}".format(str(roc_auc[i])))
        print("{:.6} (95% CI: {:.6}, {:.6})".format(str(roc_auc[i]),str(confidence_interval[0]), str(confidence_interval[1])))

        # print("(95% CI: '{:.6}'.format(confidence_interval[0]), '{:.6}'.format(confidence_interval[1])")


        # prob = df[str(i)]
        # prob = np.array(prob)
        # label = np.full_like(prob, 1)
        # fpr[i], tpr[i], _ = roc_curve(label, prob)
        # roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制AUC曲线
    plt.figure(figsize=(8, 6))
    for i in range(6):
        plt.plot(fpr[i], tpr[i], label=f'{class_dict[i]} (AUC = {str(roc_auc[i])[:6]}))')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 对角线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve of multi-parametric MRI slices ({model_name})')
    plt.legend(loc='lower right')

    plt.savefig('./picture/ROC curve of ' + model_name + '.png', format='png', dpi=300)
    png = Image.open('./picture/ROC curve of ' + model_name + '.png')
    png.save('./picture/ROC curve of ' + model_name + '.tif')
    plt.show()

# if "__name__"=="__main__":


# print(df.dtype)
# # 提取特征和标签
# X = df.drop('class_label', axis=1)  # 请替换 'class_label' 为你的标签列的名称
# y = df['class_label']
#
# # 将标签进行二值化处理
# y_bin = label_binarize(y, classes=np.unique(y))
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)
#
# # 初始化随机森林分类器
# classifier = OneVsRestClassifier(RandomForestClassifier())
#
# # 训练模型
# classifier.fit(X_train, y_train)
#
# # 获取每个类别的预测概率
# y_score = classifier.predict_proba(X_test)
#
# # 计算每个类别的ROC曲线和AUC值
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(len(np.unique(y))):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # 绘制AUC曲线
# plt.figure(figsize=(8, 6))
# for i in range(len(np.unique(y))):
#     plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
#
# plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 对角线
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Multi-Class ROC Curve')
# plt.legend(loc='lower right')
# plt.show()
