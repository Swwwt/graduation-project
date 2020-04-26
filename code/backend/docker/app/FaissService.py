from flask import Flask, request
from flask_restful import Resource, Api
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
import faiss
import numpy as np
import json


class SmallModel():
    def __init__(self):
        filename = './model/glove.6B.50d.txt'
        self.model = {}
        with open(filename, 'r') as f:
            for line in f:
                line = line.split()
                word = line[0]
                vec = [float(i) for i in line[1:]]
                vec = np.array(vec, dtype='float32')
                self.model[word] = vec
        self.vocab = list(self.model.keys())

    def __getitem__(self, x):
        if isinstance(x, str):
            return self.model[x]
        elif isinstance(x, list):
            return np.stack([self.model[i] for i in x])
        else:
            raise ValueError("Type Not Supported")


class FaissService():
    def __init__(self):
        '''
            filename = './model/GoogleNews-vectors-negative300.bin'
            self._model = KeyedVectors.load_word2vec_format(
                filename, binary=True)
            self._d = 300
        '''
        self._model = SmallModel()
        self._d = 50

        self._k = 5
        self._index = faiss.IndexFlatL2(self._d)
        print("##faiss init success", self._index.is_trained)

    def sent_to_vec(self, sentences):
        sent2vec = []
        for sentence in sentences:
            words = simple_preprocess(sentence)
            words = [word for word in words if word in self._model.vocab]
            sent2vec.append(np.mean(self._model[words], axis=0))
        return sent2vec

    def clear(self):
        self._index.reset()
        self._set([], [])

    def build(self, questions, answers):
        db = np.asarray(self.sent_to_vec(questions), dtype=np.float32)
        self.clear()
        self._index.add(db)
        self._set(questions, answers)
        print(self._index.ntotal)

    def search(self, question):
        q = np.asarray(self.sent_to_vec([question]), dtype=np.float32)
        print(self._index.ntotal)

        print("###q", q)
        D, I = self._index.search(q, self._k)
        print("###I", I)
        print("###D", D)
        ans = []
        for i, d in zip(I[0], D[0]):
            if i != -1:
                ans.append(
                    {"id": str(i), "question": self._question[i], "answer": self._answers[i], "distance": str(d)})
        print("ans", ans)
        return ans

    def _set(self, question, answers):
        self._question = question
        self._answers = answers
