# -*- coding: utf-8 -*-
from constants import *
from sklearn.externals import joblib
from sklearn.svm import SVC
import os
import numpy as np
from wdbedding import load_word2vec_model,embedding,word_segmentation

def test(model,tag):
    test_set=np.load(os.path.join(Dataset_Dir,Tag_Name[tag],"%s_train_.npz"%Tag_Name[tag]))
    y_test=test_set["arr_0"]
    test_size=y_test.shape[0]
    X_test=[]
    for i in range(test_size):
        if test_set["arr_%d"%(i+1)].shape[0]==0:
            continue
        txt_word2vec=test_set["arr_%d"%(i+1)]
        feature=sum(np.array(txt_word2vec))/len(txt_word2vec)
        X_test.append(feature)
    result=model.predict(X_test)
    correct=0
    for i,res in enumerate(result):
        if res==y_test[i]:
            correct=correct+1
    print("The precision: {:.2f}%".format((correct/test_size)*100))
    #print (model.score(X_test,y_test))

def get_txt_word2vec(word2vec_model,txt,lan):
    seg_list=word_segmentation(txt,lan)
    #print (seg_list)
    txt_embedding=[]
    for word in seg_list:
        if word in word2vec_model:
            #print (word)
            txt_embedding.append(word2vec_model[word])
    feature=np.array([])
    if (len(txt_embedding)!=0):
        feature=sum(np.array(txt_embedding))/len(txt_embedding)
    return feature
     

def test_sentence(svm_model,wd2vec_model,txt,lan):
    feature=get_txt_word2vec(wd2vec_model,txt,lan)
    if feature.shape[0]==0: 
        return 0
    else:
        feature=feature.reshape(1,-1)
    if svm_model.predict(feature)==1:
        #print ("SA ----> Positive")
        return 1
    else:
        #print ("SA ----> Negative")
        return -1

if __name__ == '__main__':
    
    #model loaded...
    
    tag=0
    clf_cn=joblib.load(os.path.join(Model_Dir,Tag_Name[tag],"%s_model_.m"%Tag_Name[tag]))
    print ("%s SVM model loaded" % Tag_Name[tag])
    wd2vec_model_cn=load_word2vec_model(tag)
    print ("%s word2vec model loaded"%Tag_Name[tag])

    #model loaded...
    txt="not null"
    while txt!="":
        txt=input("input txt: ")
        test_sentence(clf_cn,wd2vec_model_cn,txt,CN)
    
    #txt="这个人有点东西"
    #test_sentence(clf_cn,wd2vec_model_cn,txt,CN)
    #txt="it's fantastic man I shall buy it again"
    #feature=get_txt_word2vec(word2vec_model,txt,CN)
    
    #test(clf_cn,0)
    
