from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import utils


@register_criterion('squared_loss')
class SquaredLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        y_pred = model(sample['net_input']['src_tokens'])
        loss = (y_pred - sample['target']).pow(2).sum()
        sample_size = sample['net_input']['src_tokens'].size(0)
        logging_output = {'loss': utils.item(loss), 'sample_size': sample_size, 'ntokens': sample_size,
                          'nsentences': sample_size}
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        agg_output = {'loss': loss_sum/float(sample_size), 'sample_size': sample_size, 'ntokens': ntokens,
                      'nsentences': nsentences}
        return agg_output
