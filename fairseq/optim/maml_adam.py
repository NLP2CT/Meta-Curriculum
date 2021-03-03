# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.optim

from . import FairseqOptimizer, register_optimizer


@register_optimizer('maml_adam')
class MamlFairseqAdam(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self._optimizer = MamlAdam(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
        }

    def set_grad(self, grad):
        return self._optimizer.set_grad(grad)

    def get_fast_weights(self):
        return self._optimizer.get_fast_weights()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        return self._optimizer.multiply_grads(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        return self._optimizer.clip_grad_norm(max_norm)


class MamlAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MamlAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            def update_fn(i):
                p = group['params'][i]
                if p.grad is None:
                    return None
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[i]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    return p.add(-group['weight_decay'] * group['lr'], p.data)

                return p.addcdiv(-step_size, exp_avg, denom)

            group['params'] = list(map(update_fn, range(len(group['params']))))

        return loss

    def set_grad(self, grad):
        # Accumulates gradients similar to backward()
        for group in self.param_groups:
            for p, dp in zip(group['params'], grad):
                if p.grad is not None:
                    p.grad = p.grad + dp
                else:
                    p.grad = dp

    def get_fast_weights(self):
        assert len(self.param_groups) == 1
        return self.param_groups[0]['params']

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        assert len(self.param_groups) == 1
        for p in self.param_groups[0]['params']:
            if p.grad is not None:
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        assert len(self.param_groups) == 1
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], max_norm)
        else:
            return math.sqrt(sum(p.grad.data.norm()**2 for p in self.param_groups[0]['params'] if p.grad is not None))

