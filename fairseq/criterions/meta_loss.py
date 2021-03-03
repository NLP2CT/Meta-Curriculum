import gc
from copy import deepcopy
from functools import reduce
from collections import defaultdict
import ipdb

import torch

import fairseq
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.modules.learner import Learner
from fairseq.meters import AverageMeter


def copy_module(model):
    return deepcopy(model.module)


def noop(model, train_task, task_criterion):
    pass


def forward_with_dummy_batch(model, train_task, task_criterion):
    train_dataset = train_task.dataset(train_task.args.train_subset)
    max_positions = utils.resolve_max_positions(train_task.max_positions(), model.max_positions(),)
    dummy_batch = train_dataset.get_dummy_batch(num_tokens=train_task.args.max_tokens, max_positions=max_positions,
                                                src_len=train_task.args.max_tokens,
                                                tgt_len=train_task.args.max_tokens)
    dummy_batch = utils.move_to_cuda(dummy_batch)
    train_task.forward_step(sample=dummy_batch, model=model, criterion=task_criterion)


def build_single_task_loss(distributed_world_size, criterion):
    if distributed_world_size == 1:
        copy_step = deepcopy
        forward_step = noop
    else:
        copy_step = copy_module
        forward_step = forward_with_dummy_batch
    if criterion == 'maml':
        meta_loss_function = maml_loss
    elif criterion == 'reptile':
        meta_loss_function = reptile
    else:
        assert criterion == 'fomaml'
        meta_loss_function = fomaml_loss
    return ComputeLoss(copy_step=copy_step, forward_step=forward_step, meta_loss_function=meta_loss_function)


class ComputeLoss(object):
    def __init__(self, copy_step, forward_step, meta_loss_function):
        self.copy_step = copy_step
        self.forward_step = forward_step
        self.meta_loss_function = meta_loss_function

    def __call__(self, train_task, model):
        return self.meta_loss_function(train_task=train_task, model=model, copy_fn=self.copy_step,
                                       forward_fn=self.forward_step)


class MetaLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.compute_loss = build_single_task_loss(distributed_world_size=args.distributed_world_size,
                                                   criterion=args.criterion)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss = 0.0
        downstream_loss = 0.0
        bleu_stats = utils.get_zero_bleu_stats()
        intermediate_loss = defaultdict(lambda: AverageMeter())
        intermediate_bleu = defaultdict(lambda: AverageMeter())
        for train_task in sample:
            # ipdb.set_trace()
            # print("| [Meta-Loss] Compute loss using {}".format(str(set(train_task.user_data_frame.task_group.values))))
            single_loss, single_loss_stats = self.compute_loss(train_task=train_task, model=model)
            single_downstream_loss, single_bleu_stats, single_intermediate_loss, single_intermediate_bleu = \
                single_loss_stats
            loss += single_loss
            downstream_loss += single_downstream_loss
            bleu_stats = utils.reduce_bleu_stats(bleu_stats, single_bleu_stats)
            for key, value in single_intermediate_loss.items():
                intermediate_loss[key] = utils.reduce_average_meter(intermediate_loss[key], value)
            for key, value in single_intermediate_bleu.items():
                intermediate_bleu[key] = utils.reduce_average_meter(intermediate_bleu[key], value)
        sample_size = len(sample)
        logging_output = {'loss': utils.item(loss.data), 'sample_size': sample_size, 'ntokens': sample_size,
                          'nsentences': sample_size, 'downstream_loss': downstream_loss,
                          'bleu': bleu_stats, 'intermediate_loss': intermediate_loss,
                          'intermediate_bleu': intermediate_bleu}
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        downstream_loss = sum(log.get('downstream_loss', 0) for log in logging_outputs)
        bleu_stats = reduce(utils.reduce_bleu_stats,
                            [log.get('bleu', utils.get_zero_bleu_stats()) for log in logging_outputs])
        intermediate_loss = defaultdict(lambda: AverageMeter())
        all_loss_dicts = [log.get('intermediate_loss', defaultdict(lambda: AverageMeter())) for log in logging_outputs]
        # Aggregate
        for single_dict in all_loss_dicts:
            for key, value in single_dict.items():
                intermediate_loss[key] = utils.reduce_average_meter(intermediate_loss[key], value)
        # same for bleu
        intermediate_bleu = defaultdict(lambda: AverageMeter())
        all_bleu_dicts = [log.get('intermediate_bleu', defaultdict(lambda: AverageMeter()))
                          for log in logging_outputs]
        # Aggregate
        for single_dict in all_bleu_dicts:
            for key, value in single_dict.items():
                intermediate_bleu[key] = utils.reduce_average_meter(intermediate_bleu[key], value)
        agg_output = {'loss': loss_sum / float(sample_size), 'sample_size': sample_size, 'ntokens': ntokens,
                      'nsentences': nsentences, 'downstream_loss': downstream_loss / float(sample_size),
                      'bleu': bleu_stats, **intermediate_loss, **intermediate_bleu}
        return agg_output


def get_number_of_tensors():
    # prints currently alive Tensors and Variables
    tensor_list = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor_list.append(obj)
        except:
            pass
#    print('number of tensors: ', len(tensor_list))


def get_loss_stats(sgd_dict):
    downstream_loss = sgd_dict['valid_losses'][0]
    bleu_stats = sgd_dict['valid_stats'][0]['bleu']
    batch_info = sgd_dict['batch_info']
    intermediate_loss = defaultdict(lambda: AverageMeter())
    intermediate_bleu = defaultdict(lambda: AverageMeter())
    for valid_stats in batch_info:
        valid_stats = valid_stats[0]
        num_updates = valid_stats['num_updates']
        loss_value = valid_stats['loss']
        loss_key = 'loss_' + str(num_updates)
        intermediate_loss[loss_key].update(loss_value)
        bleu_value = valid_stats['bleu'].avg
        bleu_key = 'bleu_' + str(num_updates)
        intermediate_bleu[bleu_key].update(bleu_value)
    return downstream_loss, bleu_stats, intermediate_loss, intermediate_bleu


def reptile(train_task, model, copy_fn, forward_fn):
    gc.collect()
    torch.cuda.empty_cache()
    args = train_task.args
    task_criterion = train_task.build_criterion(args)
    # SGD step
    fine_tuned_model = copy_fn(model)
    sgd_dict = utils.sgd(task=train_task, args=args, model=fine_tuned_model, criterion=task_criterion,
                         train_function=utils.train)
    loss_stats = get_loss_stats(sgd_dict=sgd_dict)
    forward_fn(model=model, train_task=train_task, task_criterion=task_criterion)
    meta_loss = 0.0
    # The values of the parameters after fine-tuning is our target
    for t, p in zip(fine_tuned_model.parameters(), model.parameters()):
        meta_loss += (t.detach() - p).pow(2).sum()
    return meta_loss, loss_stats


def fomaml_loss(train_task, model, copy_fn, forward_fn):
    args = train_task.args
    task_criterion = train_task.build_criterion(args)
    # SGD step
    fine_tuned_model = copy_fn(model)
    sgd_dict = utils.sgd(task=train_task, args=args, model=fine_tuned_model, criterion=task_criterion,
                         train_function=utils.train)
    loss_stats = get_loss_stats(sgd_dict=sgd_dict)
    forward_fn(model=model, train_task=train_task, task_criterion=task_criterion)
    # Meta-Loss computation
    subset = args.valid_subset.split(',')[0]
    trainer = sgd_dict['trainer']
    trainer.optimizer.zero_grad()
    task_loss = utils.forward_and_accumulate_gradient(args=args, trainer=trainer, task=train_task, epoch_itr=sgd_dict['epoch_itr'], subset=subset)
    for t, p in zip(fine_tuned_model.parameters(), model.parameters()):
        p.grad = t.grad
    meta_loss = torch.ones(1, requires_grad=True) * task_loss
    if trainer.cuda:
        meta_loss = meta_loss.cuda()
    return meta_loss, loss_stats


def maml_loss(train_task, model, copy_fn, forward_fn):
    is_training = model.training
#    print('model.training: ', is_training)
    fine_tuned_model = copy_fn(model)
    args = train_task.args
    task_criterion = train_task.build_criterion(args)
    # Start of MAML loss computation
    gc.collect()
    torch.cuda.empty_cache()
    maml_args = deepcopy(args)

    if is_training:
        # Keep track of the computational graph only at training time
        maml_args.optimizer = 'maml_' + args.optimizer
        learner = Learner(fine_tuned_model)
        maml_task = fairseq.tasks.maml_task.MamlTask(base_task=train_task, learner=learner)
        maml_model = model
    else:
        # Faster version for validation
        maml_task = train_task
        maml_model = fine_tuned_model

    sgd_dict = utils.sgd(task=maml_task, args=maml_args, model=maml_model, criterion=task_criterion,
                         train_function=utils.train)

    maml_trainer = sgd_dict['trainer']

    loss_stats = get_loss_stats(sgd_dict=sgd_dict)
    forward_fn(model=model, train_task=train_task, task_criterion=task_criterion)

    subset = maml_args.valid_subset.split(',')[0]

    if is_training:
        meta_loss = utils.forward(args=maml_args, trainer=maml_trainer, task=maml_task, epoch_itr=sgd_dict['epoch_itr'],
                                  subset=subset)
    else:
        meta_loss = torch.Tensor([0.0])
    print('meta_loss: ', meta_loss)
    maml_trainer.optimizer.zero_grad()
    # Make sure we clear the original model gradients!
    for p in model.parameters():
        p.grad = None
#    print('meta_loss: ', meta_loss)
    del maml_trainer
    gc.collect()
    torch.cuda.empty_cache()
    return meta_loss, loss_stats


@register_criterion('reptile')
class ReptileLoss(MetaLoss):

    def __init__(self, args, task):
        super().__init__(args, task)


@register_criterion('maml')
class MamlLoss(MetaLoss):

    def __init__(self, args, task):
        super().__init__(args, task)


@register_criterion('fomaml')
class FoMamlLoss(MetaLoss):

    def __init__(self, args, task):
        super().__init__(args, task)
