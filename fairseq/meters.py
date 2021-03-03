# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import time
import math

from fairseq import utils
from fairseq import bleu


class Meter(object):

    def reset(self):
        pass

    def update(self, val, n=1):
        pass

    @property
    def avg(self):
        pass

    @property
    def std(self):
        return 0.0

class AverageMeter(Meter):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.sum_square = 0

    def reset(self):
        self.sum = 0
        self.count = 0
        self.sum_square = 0

    def update(self, val, n=1):
        if isinstance(val, AverageMeter):
            reduced_meter: AverageMeter = utils.reduce_average_meter(self, val)
            self.sum = reduced_meter.sum
            self.count = reduced_meter.count
            self.sum_square = reduced_meter.sum_square
        else:
            self.sum += val * n
            self.count += n
            self.sum_square = self.sum_square + (val * val) * n

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    @property
    def std(self):
        expected_sum_square = self.sum_square / self.count
        expected_sum = self.avg
        return math.sqrt(expected_sum_square - expected_sum * expected_sum)


class ConcatentateMeter(Meter):
    def __init__(self, lowercase=False):
        self.scorer = bleu.SacrebleuScorer(lowercase=lowercase)
        self.target_sum = []
        self.hypo_sum = []
        self.count = 0

    def reset(self):
        self.target_sum = []
        self.hypo_sum = []
        self.count = 0

    def update(self, val, n=1):
        self.target_sum += val[0] * n
        self.hypo_sum += val[1] * n
        self.count += n
        # TODO compute corpus bleu here

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        # Compute the corpus level BLEU
        self.scorer.sys = self.hypo_sum
        self.scorer.ref = self.target_sum
        return self.scorer.score()


class BleuMeter(Meter):
    def __init__(self):
        self.correct, self.total, self.sys_len, self.ref_len = utils.get_zero_bleu_stats()
        # TODO handle lowercase
        self.scorer = bleu.SacrebleuScorer(lowercase=False)

    def reset(self):
        self.correct, self.total, self.sys_len, self.ref_len = utils.get_zero_bleu_stats()

    def update(self, val, n=1):
        # val will be a namedtuple
        # We need to reduce
        for _ in range(n):
            self.correct = utils.reduce_lists(self.correct, val.correct)
            self.total = utils.reduce_lists(self.total, val.total)
            self.sys_len += val.sys_len
            self.ref_len += val.ref_len

    @property
    def avg(self):
        # We have the sufficient statistics, just compute the BLEU score
        return self.scorer.compute_bleu(correct=self.correct, total=self.total, sys_len=self.sys_len,
                                        ref_len=self.ref_len).score


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""
    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(Meter):
    """Computes the sum/avg duration of some event in seconds"""
    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n
