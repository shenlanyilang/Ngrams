# -*- coding: utf-8 -*-
import math
from typing import List,Dict,Set
import numpy as np



class Smoothing(object):
    def __init__(self):
        pass

    def update_model(self,ngram_model):
        raise NotImplementedError

class NGRAMS(object):
    def __init__(self, n=3):
        self.N = n
        self.root_node = NgramNode()
        self.vocab = set()
        self.unseen_prob = 0.

    def get_count_of_counts(self,arr):
        self.root_node.get_count_of_counts(arr, self.N)

    def add_sentences(self,corpus:List[List[str]]):
        '''
        add corpus to construct the language model
        :param corpus:
        :return:
        '''
        for sentence in corpus:
            for i in range(0, len(sentence) - self.N + 1):
                ngram = sentence[i:i+self.N]
                self.vocab.update(ngram)
                self.root_node.add_ngram(ngram)

    def get_ngrams_prob(self, ngram:List[str])->float:
        '''
        get ngram's probability
        :param ngram:
        :param n:
        :return:
        '''
        prob = self.root_node.get_ngrams_prob(ngram)
        if prob == 0.:
            prob = self.unseen_prob
        return prob

    def sententce_perplexity(self,sentence:List[str])->float:
        '''
        get sentence perplexity
        :param sentence:
        :return:
        '''
        if len(sentence) < self.N:
            print('sentence length less than {}'.format(self.N))
            return 0.
        sum_log_per = 0
        total_cnt = 0
        for i in range(0, len(sentence) - self.N + 1):
            ngram = sentence[i:self.N]
            prob = self.get_ngrams_prob(ngram)
            sum_log_per += -math.log(prob)
            total_cnt += 1
        return math.exp(sum_log_per / total_cnt)

    def corpus_perplexity(self,corpus:List[List[str]]):
        total_cnt = 0
        sum_log_per = 0
        for sentence in corpus:
            for i in range(0, len(sentence) - self.N + 1):
                ngram = sentence[i:i + self.N]
                prob = self.get_ngrams_prob(ngram)
                sum_log_per += -math.log(prob)
                total_cnt += 1
        return math.exp(sum_log_per / total_cnt)

    def add_smoothing(self, smoothing_method:Smoothing):
        '''
        add smoothing method to the model
        :param smoothing_method:
        :return:
        '''
        smoothing_method.update_model(self)

    def remove_unknown_words(self, vocabulary:Set[str]):
        '''
        replace words whose frequency less than threshold with a unify word
        :param threshold:
        :return:
        '''
        self.root_node.remove_unknown_words(vocabulary)

    def adjust_probobility(self, count_array, prob):
        self.root_node.adjust_probability(count_array,self.N, prob, len(self.vocab))



class NgramNode(object):
    def __init__(self,symbol=''):
        self.symbol:str = symbol
        self.children:Dict[str, NgramNode] = dict()
        self.count = 0
        self.sum = 0
        self.prob = 0
        self.unknown:NgramNode = None
        self.unseen_prob = 0.

    def get_count_of_counts(self,arr,n):
        if n == 1:
            for symbol, child in self.children.items():
                count = child.count
                if count >= len(arr):
                    arr.extend([0] * (count - len(arr) + 1))
                arr[count] += 1
        else:
            for symbol,child in self.children.items():
                child.get_count_of_counts(arr, n-1)

    def add_ngram(self,ngram:List[str]):
        '''
        add ngram to the node
        :param ngram:
        :return:
        '''
        if len(ngram) == 0:
            return
        self.count += 1
        self.sum += 1
        word = ngram[0]
        if word not in self.children:
            self.children[word] = NgramNode()
        self.children[word].count += 1

        for word, child in self.children.items():
            child.prob = child.count / self.sum
        self.children[word].add_ngram(ngram[1:])


    def get_ngrams_prob(self, ngrams:List[str])->float:
        assert len(ngrams) >= 1
        if len(ngrams) == 1:
            word = ngrams[0]
            if word in self.children:
                return self.children[word].prob
            if self.unknown:
                return self.unknown.prob
            return self.unseen_prob
        word = ngrams[0]
        if word in self.children:
            return self.children[word].get_ngrams_prob(ngrams[1:])
        return 0.

    def add_pseudo_count(self, alpha, vsize):
        '''
        add pseudo count for ngrams, used for smoothing
        :param alpha:
        :param vsize:
        :return:
        '''
        self.count += alpha
        self.sum += alpha * vsize
        if self.unknown:
            self.unknown.add_pseudo_count(alpha, vsize)
            self.unknown.prob = self.unknown.count / self.sum
        for symbol, child in self.children.items():
            child.add_pseudo_count(alpha, vsize)
            child.prob = child.count / self.sum
        self.unseen_prob = alpha / self.sum

    def remove_unknown_words(self, vocabulary:Set[str]):
        '''
        combine words together which not in vocabulary
        :param vocabulary:
        :return:
        '''
        unknown_symbols = set()
        for symbol in self.children:
            if symbol not in vocabulary:
                unknown_symbols.add(symbol)
        for symbol in unknown_symbols:
            child = self.children[symbol]
            del self.children[symbol]
            self.unknown.add_children(child.children)
        for child in self.children.values():
            child.remove_unknown_words(vocabulary)
        for child in self.unknown.children.values():
            child.remove_unknown_words(vocabulary)

    def add_children(self, children:Dict):
        '''
        add children to the nodes's already existing children
        :param children:
        :return:
        '''
        for symbol in children.keys():
            child = children[symbol]
            self.count += child.count
            self.sum += child.count
            if symbol not in self.children:
                self.children[symbol] = child
            else:
                self.children[symbol].merge(child)

    def merge(self, child):
        '''
        merge a node which has the same symbol as the node
        :param child:
        :return:
        '''
        assert self.symbol == child.symbol
        self.count += child.count
        self.sum += child.count
        self.add_children(child.children)

    def adjust_probability(self,count_array, n, prob, vsize):
        sum = 0.
        if n == 1:
            for symbol, child in self.children.items():
                r = child.count
                if r < 5:
                    child.count = (r + 1) * count_array[r+1] / count_array[r]
                    sum += child.count
                else:
                    child.count = 5
                    sum += child.count
            for symbol, child in self.children.items():
                child.prob = (1-prob) * child.count / sum
            self.unseen_prob = prob / (vsize - len(self.children))


class LaplaceSmoothing(Smoothing):
    '''
    Laplace smoothing method
    '''
    def __init__(self,alpha=0.01):
        self.alpha = alpha
        super(LaplaceSmoothing, self).__init__()

    def update_model(self,ngram_model:NGRAMS):
        ngram_model.root_node.add_pseudo_count(self.alpha, len(ngram_model.vocab))


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
        counts_modify = [0] * len(count_array)
        r= []
        c = []
        for i,num in enumerate(count_array[1:], start=1):
            if num != 0:
                r.append(i)
                c.append(num)
        a = np.zeros(shape=(2,2))
        y = np.zeros(shape=(1,2))
        for i, num in enumerate(r):
            xt = math.log1p(num)
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
            w = np.dot(a,y)
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