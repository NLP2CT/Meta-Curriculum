from fairseq.data import Dictionary
from fairseq.data.meta_dataset import MetaFairseqDataset
from . import FairseqTask, register_task
import torch
import os

# Represents a sine task
@register_task('meta')
class MetaTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default='en', metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default='de', metavar='TARGET',
                            help='target language')

    def __init__(self, args, meta_train_tasks, meta_dev_tasks, meta_test_tasks, src_dict, tgt_dict):
        self.datasets = {}
        self.args = args
        self.meta_train_tasks = meta_train_tasks
        self.meta_dev_tasks = meta_dev_tasks
        self.meta_test_tasks = meta_test_tasks
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, meta_train_tasks, meta_dev_tasks, meta_test_tasks, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        return cls(args=args, meta_train_tasks=meta_train_tasks, meta_dev_tasks=meta_dev_tasks,
                   meta_test_tasks=meta_test_tasks, src_dict=src_dict, tgt_dict=tgt_dict)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split == self.args.train_subset:
            self.datasets[split] = MetaFairseqDataset(split=split, meta_tasks=self.meta_train_tasks)
        elif split == self.args.valid_subset.split(',')[0]:
            self.datasets[split] = MetaFairseqDataset(split=split, meta_tasks=self.meta_dev_tasks)
        else:
            self.datasets[split] = MetaFairseqDataset(split=split, meta_tasks=self.meta_test_tasks)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        assert False

    def valid_step(self, sample, model, criterion, optimizer=None):
        model.eval()
        with torch.enable_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
