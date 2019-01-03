中文情感分析  SVM分类模型

preprocess.py 处理数据，划分训练集与测试集，生成npz文件
wdbedding.py 词向量相关，包括载入模型（模型在word_embedding文件夹），将文本中词转换为对应词向量
train.py 由训练数据训练svm model，生成model在model文件夹
constants.py 声明一些全局参数
test.py 对输入文本判断情感
analysis.py 读取xml文件，输出标明各个句子情感极性的xml文件