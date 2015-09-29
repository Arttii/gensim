#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jan Zikes, Radim Rehurek
# Copyright (C) 2014 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging

from gensim import utils
from gensim.models.ldamodel import LdaModel, LdaState
from six.moves import queue, xrange
from multiprocessing import Pool, Queue, cpu_count
from _functools import partial
logger = logging.getLogger(__name__)

from pyspark.accumulators import AccumulatorParam


class LDAStateAccumParam(AccumulatorParam):

    def zero(self, value):
        return value

    def addInPlace(self, val1, val2):
        val1.merge(val2)
        return val1


class LdaSpark(LdaModel):

    def __init__(self, id2word, corpus=None, num_topics=100, workers=None,
                 chunksize=2000, passes=100, batch=False, alpha='symmetric',
                 eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50,
                 gamma_threshold=0.001):

        self.workers = max(1, cpu_count() - 1) if workers is None else workers
        self.batch = batch

        if alpha == 'auto':
            raise NotImplementedError(
                "auto-tuning alpha not implemented in multicore LDA; use plain LdaModel.")

        super(LdaSpark, self).__init__(corpus=corpus, num_topics=num_topics,
                                       id2word=id2word, chunksize=chunksize, passes=passes, alpha=alpha, eta=eta,
                                       decay=decay, offset=offset, eval_every=eval_every, iterations=iterations,
                                       gamma_threshold=gamma_threshold)

    def update(self, sc, corpus):
        corpus.cache()
        lencorpus = corpus.count()
        # rho is the "speed" of updating, decelerating over time
        rho = lambda: pow(
            self.offset + self.num_updates / self.chunksize, -self.decay)

        self.state.numdocs += lencorpus

        corpus = corpus.repartition(
            min(self.chunksize * self.workers, lencorpus))

        def s_e_step(worker_lda, state, chunk):
            worker_lda.state.reset()
            worker_lda.do_estep(chunk)
            state.add(worker_lda.state)
            del worker_lda

        for pass_ in xrange(self.passes):
            other = sc.accumulator(
                LdaState(self.eta, self.state.sstats.shape), LDAStateAccumParam())
            partitioned_e_step = partial(s_e_step, self, other)
            corpus.foreachPartition(partitioned_e_step)

            self.do_mstep(rho(), other.value)
            other.value.reset()
            # process_result_queue
