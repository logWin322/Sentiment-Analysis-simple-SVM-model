# -*- coding: utf-8 -*-
from test import test_sentence
from wdbedding import load_word2vec_model
from constants import *
from sklearn.externals import joblib
try:
    import xml.etree.cElementTree as ET 
except ImportError:
    import xml.etree.ElementTree as ET
import os


def tagging(txt):
    for char in txt:
        if not ((char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z') or (char >= '0' and char <= '9') or char in " ,.<>/?\\!@#$%^&*()-_+=`~\t;:\'\"[]{}|"):
            return CN
    return EN

def analysis(input_file,output_file):
    xml_tree=ET.parse(input_file)
    xml_root=xml_tree.getroot()
    tag=0
    #wd2vec_model = list(map(lambda x: load_word2vec_model(x), Languages))
    wd2vec_model_cn=load_word2vec_model(tag)
    #svm_model=list(map(lambda x:joblib.load(os.path.join(Model_Dir,Tag_Name[x],"%s_model_.m"%Tag_Name[x])),Languages))
    clf_cn=joblib.load(os.path.join(Model_Dir,Tag_Name[tag],"%s_model_.m"%Tag_Name[tag]))
    for xml_node in xml_root:
        txt=xml_node.text
        if txt[-1] == '\n':
            txt = txt[:-1]
        tag = tagging(txt)
        xml_node.set('polarity',"%d"%test_sentence(clf_cn,wd2vec_model_cn,txt,tag))
    xml_tree.write(output_file,encoding='utf-8')
    print("anlysis is done.")
    
if __name__ == "__main__":
    input_file=input("input filename: ")
    output_file=input("output filename: ")
    analysis(os.path.join(Dataset_Dir,input_file),os.path.join(Dataset_Dir,output_file))
    