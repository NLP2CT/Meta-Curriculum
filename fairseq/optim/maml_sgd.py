# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.optim
import math

from . import FairseqOptimizer, register_optimizer


class TorchMamlSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super(TorchMamlSGD, self).__init__(*args, **kwargs)

    def set_grad(self, grad):
        # Accumulates gradients similar to backward()
        for group in self.param_groups:
            for p, dp in zip(group['params'], grad):
                if p.grad is not None:
                    p.grad = p.grad + dp
                else:
                    p.grad = dp

    def step(self, closure=None):
        """Performs a single optimization step. This step doesn't modify in-place, instead it creates a new variable, to
        keep track of the computational graph

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            def update_fn(p):
                if p.grad is None:
                    return None
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                return p.add(-group['lr'], d_p)

            group['params'] = list(map(update_fn, group['params']))

        return loss

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

@register_optimizer('maml_sgd')
class MamlSGD(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self._optimizer = TorchMamlSGD(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--momentum', default=0.0, type=float, metavar='M',
                            help='momentum factor')
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
            'momentum': self.args.momentum,
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
