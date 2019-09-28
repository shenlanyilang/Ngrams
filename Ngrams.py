# -*- coding: utf-8 -*-
import math
from typing import List,Dict,Set


class NGRAMS(object):
    def __init__(self, n=3):
        self.N = n
        self.root_node = NgramNode()
        self.vocab = set()
        self.unseen_prob = 0.

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

    def add_smoothing(self, smoothing_method):
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



class NgramNode(object):
    def __init__(self,symbol=''):
        self.symbol:str = symbol
        self.children:Dict[str, NgramNode] = dict()
        self.count = 0
        self.sum = 0
        self.prob = 0
        self.unknown:NgramNode = None
        self.unseen_prob = 0.


    def add_ngram(self,ngram:List[str]):
        '''
        add ngram to the node
        :param ngram:
        :return:
        '''
        if len(ngram) == 0:
            return
        self.count += 1
        word = ngram[0]
        if word not in self.children:
            self.children[word] = NgramNode()
        self.children[word].add_ngram(ngram[1:])

    def get_ngrams_prob(self, ngrams:List[str])->float:
        assert len(ngrams) >= 1
        if len(ngrams) == 1:
            word = ngrams[0]
            if word in self.children:
                return self.children[word].count / self.sum
            if self.unknown:
                return self.unknown.count / self.sum
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
        for symbol, child in self.children.items():
            child.add_pseudo_count(alpha, vsize)

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

class LaplaceSmoothing(object):
    '''
    Laplace smoothing method
    '''
    def __init__(self,alpha=0.01):
        self.alpha = alpha

    def update_model(self,ngram_model:NGRAMS):
        ngram_model.root_node.add_pseudo_count(self.alpha, len(ngram_model.vocab))
