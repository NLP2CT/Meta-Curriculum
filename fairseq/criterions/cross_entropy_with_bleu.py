# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
from functools import reduce
import torch

from fairseq import utils
from fairseq import bleu
from fairseq.meters import StopwatchMeter, TimeMeter

from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy_with_bleu')
class CrossEntropyWithBleuCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.task = task
        self.generator = task.build_generator(args)

    @staticmethod
    def add_args(parser):
        from fairseq.options import add_generation_args
        """Add criterion-specific arguments to the parser."""
        add_generation_args(parser)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        # Only evalute bleu if not training
        if model.training:
            bleu_stats = utils.get_zero_bleu_stats()
        else:
            use_cuda = torch.cuda.is_available() and not self.args.cpu
            if self.args.sacrebleu:
                if hasattr(self.args, 'lowercase'):
                    scorer = bleu.SacrebleuScorer(lowercase=self.args.lowercase)
                else:
                    scorer = bleu.SacrebleuScorer()
            else:
                scorer = bleu.Scorer(self.task.target_dictionary.pad(), self.task.target_dictionary.eos(),
                                     self.task.target_dictionary.unk())
            gen_timer = StopwatchMeter()
            wps_meter = TimeMeter()
            utils.run_inference_on_sample(sample=sample, use_cuda=use_cuda, args=self.args, gen_timer=gen_timer,
                                          task=self.task, generator=self.generator, model=model,
                                          tgt_dict=self.task.target_dictionary, align_dict=None, subset=None,
                                          src_dict=self.task.source_dictionary, scorer=scorer, wps_meter=wps_meter)
            result_string = scorer.result_string()
            bleu_stats = utils.BleuStatistics(correct=result_string.counts, total=result_string.totals,
                                              sys_len=result_string.sys_len, ref_len=result_string.ref_len)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'bleu': bleu_stats,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        bleu_stats = reduce(utils.reduce_bleu_stats,
                            [log.get('bleu', utils.get_zero_bleu_stats()) for log in logging_outputs])
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'bleu': bleu_stats,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
