# -*- coding: utf-8 -*-
from smoothing import Smoothing
from Ngrams import NGRAMS


class LaplaceSmoothing(Smoothing):
    '''
    Laplace smoothing method
    '''

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        super(LaplaceSmoothing, self).__init__()

    def update_model(self, ngram_model: NGRAMS):
        ngram_model.root_node.add_pseudo_count(self.alpha, len(ngram_model.vocab))
        ngram_model.unseen_prob = 1. / len(ngram_model.vocab)
