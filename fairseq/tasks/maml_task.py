import torch


class MamlTask(object):
    def __init__(self, base_task, learner):
        self.base_task = base_task
        self.learner = learner
        self.target_dictionary = base_task.target_dictionary

    def build_generator(self, args):
        return self.base_task.build_generator(args)

    def load_dataset(self, split, **kwargs):
        return self.base_task.load_dataset(split, **kwargs)

    def dataset(self, split):
        return self.base_task.dataset(split)

    def max_positions(self):
        return self.base_task.max_positions()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_batch_iterator(
            self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
            ignore_invalid_inputs=False, required_batch_size_multiple=1,
            seed=1, num_shards=1, shard_id=0, num_workers=0,
    ):
        return self.base_task.get_batch_iterator(dataset, max_tokens, max_sentences, max_positions,
                                                 ignore_invalid_inputs, required_batch_size_multiple, seed, num_shards,
                                                 shard_id, num_workers,)

    def forward_step(self, sample, model, criterion, optimizer):
        model.eval()
        learner = self.learner

        class LearnerModel(object):

            def get_targets(self, sample, net_output):
                return model.get_targets(sample=sample, net_output=net_output)

            def get_normalized_probs(self, net_output, log_probs):
                return model.get_normalized_probs(net_output=net_output, log_probs=log_probs)

            def __call__(self, *args, **kwargs):
                return learner.functional(optimizer.get_fast_weights(), training=False, *args, **kwargs)

        loss, sample_size, logging_output = criterion(LearnerModel(), sample)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, optimizer):
        model.eval()
        with torch.no_grad():
            learner = self.learner

            class LearnerModel(object):

                def get_targets(self, sample, net_output):
                    return model.get_targets(sample=sample, net_output=net_output)

                def get_normalized_probs(self, net_output, log_probs):
                    return model.get_normalized_probs(net_output=net_output, log_probs=log_probs)

                def __call__(self, *args, **kwargs):
                    return learner.functional(optimizer.get_fast_weights(), training=False, *args, **kwargs)

            loss, sample_size, logging_output = criterion(LearnerModel(), sample)
        return loss, sample_size, logging_output

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        return self.base_task.aggregate_logging_outputs(logging_outputs, criterion)

    def grad_denom(self, sample_sizes, criterion):
        return self.base_task.grad_denom(sample_sizes, criterion)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        learner = self.learner

        class LearnerModel(object):

            def get_targets(self, sample, net_output):
                return model.get_targets(sample=sample, net_output=net_output)

            def get_normalized_probs(self, net_output, log_probs):
                return model.get_normalized_probs(net_output=net_output, log_probs=log_probs)

            def __call__(self, *args, **kwargs):
                return learner.functional(optimizer.get_fast_weights(), training=True, *args, **kwargs)

        loss, sample_size, logging_output = criterion(LearnerModel(), sample)
        if ignore_grad:
            loss *= 0
        # This is a replacement for backward
        grad = torch.autograd.grad(loss, optimizer.get_fast_weights())
        optimizer.set_grad(grad)
        return loss, sample_size, logging_output

    def update_step(self, num_updates):
        return self.base_task.update_step(num_updates)


