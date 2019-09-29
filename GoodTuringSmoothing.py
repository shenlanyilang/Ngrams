# -*- coding: utf-8 -*-
from typing import List
from smoothing import Smoothing
from Ngrams import NGRAMS
import numpy as np
import math


class GoodTuringSmoothing(Smoothing):
    def __init__(self):
        super(GoodTuringSmoothing, self).__init__()
        pass

    def update_model(self,ngram_model:NGRAMS):
        self.set_probability(ngram_model)

    def count_array(self, ngram_model:NGRAMS)->List[int]:
        arr = []
        ngram_model.get_count_of_counts(arr)
        arr.extend([0])
        return arr

    def linear_regression_modify(self,count_array)->List[float]:
        counts_modify = [0.] * len(count_array)
        r= []
        c = []
        for i,num in enumerate(count_array[1:], start=1):
            if num != 0:
                r.append(i)
                c.append(num)
        a = np.zeros(shape=(2,2))
        y = np.zeros(shape=(2,))
        for i, num in enumerate(r):
            xt = math.log(num)
            if i == 0:
                rt = math.log(c[i])
            else:
                if i == len(r) - 1:
                    rt = math.log((1.0 * c[i])/(r[i] - r[i-1]))
                else:
                    rt = math.log((2.0 * c[i]) / (r[i+1] - r[i-1]))
            a[0,0] += 1.0
            a[0,1] += xt
            a[1,0] += xt
            a[1,1] += xt * xt
            y[0] += rt
            y[1] += rt * xt
        try:
            a = np.linalg.inv(a)
            w = np.dot(a,y.reshape(2,1))
            w0 = w.flatten()[0]
            w1 = w.flatten()[1]
            for i in range(1,len(count_array)):
                counts_modify[i] = math.exp(math.log(i) * w1 + w0)
        except:
            pass
        return counts_modify

    def set_probability(self,ngram_model:NGRAMS)->None:
        origin_count_array = self.count_array(ngram_model)
        count_array_modify = self.linear_regression_modify(origin_count_array)
        sum = 0
        for i in range(1,len(count_array_modify)):
            sum += i * count_array_modify[i]
        ngram_model.adjust_probobility(count_array_modify, count_array_modify[1] / sum)
        ngram_model.unseen_prob = 1. / len(ngram_model.vocab)
