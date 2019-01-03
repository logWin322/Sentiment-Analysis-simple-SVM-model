import gensim, logging
from multiprocessing import Queue
import os

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),'r',encoding='UTF-8'):
                yield line.split()

if __name__ == '__main__':
    sentences = MySentences('training-monolingual.tokenized.shuffled')
    model = gensim.models.Word2Vec(sentences)
    model.save('word2vec_en_training.txt')
    