import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report # 结果评估

class Myreport:
    def __init__(self):
        self.__confusion = None

    def __statistics_confusion(self,y_true,y_predict):
        self.__confusion = np.zeros((6, 6))
        for i in range(y_true.shape[0]):
            self.__confusion[y_predict[i]][y_true[i]] += 1

    def __cal_Acc(self):
        return np.sum(self.__confusion.diagonal()) / np.sum(self.__confusion)

    def __cal_Pc(self):
        return self.__confusion.diagonal() / np.sum(self.__confusion, axis=1)

    def __cal_Rc(self):
        return self.__confusion.diagonal() / np.sum(self.__confusion, axis=0)

    def __cal_F1score(self,PC,RC):
        return 2 * np.multiply(PC, RC) / (PC + RC)

    # def __cal_

    def report(self,y_true,y_predict,classNames):
        self.__statistics_confusion(y_true,y_predict)
        Acc = self.__cal_Acc()
        Pc = self.__cal_Pc()
        Rc = self.__cal_Rc()
        F1score = self.__cal_F1score(Pc,Rc)
        str = "Class Name\t\tprecision\t\trecall\t\tf1-score\n"
        for i in range(len(classNames)):
           str += f"{classNames[i]}   \t\t\t{format(Pc[i],'.2f')}   \t\t\t{format(Rc[i],'.2f')}" \
                  f"   \t\t\t{format(F1score[i],'.2f')}\n"
        str += f"accuracy is {format(Acc,'.2f')}"
        return str


if __name__ == "__main__":


    last_column_predict = {}
    last_column_true = {}



    for i in range(6):
        filepath = os.path.join('./predict_result0', str(i) + '.csv')
        df = pd.read_csv(filepath)
        last_column_predict[i] = df.iloc[:, -1].values
        # last_column_predict[i] = np.squeeze(df.iloc[:, -1].values)
        last_column_true[i] = np.full_like(last_column_predict[i],i)

    y_predict = np.concatenate([array for array in last_column_predict.values()]).astype(int)
    y_true = np.concatenate([array for array in last_column_true.values()]).astype(int)

    kappa_score = cohen_kappa_score(y_true, y_predict)
    # myreport = Myreport()
    # print("=====自己实现的结果=====")
    # print(myreport.report(y_true=y_true, y_predict=y_predict,classNames=['0','1','2','3','4','5']))
    print("=====使用sklrean的结果=====")
    print(classification_report(y_true, y_predict, target_names=['0','1','2','3','4','5'], digits=4))
    print(kappa_score)

