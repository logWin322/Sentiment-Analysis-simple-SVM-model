# -*- coding: utf-8 -*-
import jieba
import os
from wdbedding import load_word2vec_model, embedding
from constants import *
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

def train_svm_model(tag,c,ker):
    train_set = np.load(os.path.join(Dataset_Dir,Tag_Name[tag],"%s_train_.npz"%Tag_Name[tag]))
    y_trained=train_set["arr_0"]
    train_size=y_trained.shape[0]
    X_trained=[]
    
    for i in range(train_size):
        if train_set["arr_%d"%(i+1)].shape[0]==0:
            continue
        txt_word2vec=train_set["arr_%d"%(i+1)]
        feature=sum(np.array(txt_word2vec))/len(txt_word2vec)
        X_trained.append(feature)
    clf=SVC(C=c,probability=True,kernel=ker)
    clf.fit(X_trained,y_trained)
    joblib.dump(clf,os.path.join(Model_Dir,Tag_Name[tag],"%s_model_.m"%Tag_Name[tag]))
    print("%s SVM model trained and saved successfully"%Tag_Name[tag])
    

if __name__ == '__main__':
    #train_svm_model(EN,0.5,'rbf')
    train_svm_model(CN,1.75,'linear')


