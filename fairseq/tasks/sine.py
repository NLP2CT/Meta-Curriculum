from fairseq.data import Dictionary
from fairseq.data.sine_dataset import SineFairseqDataset
from . import FairseqTask, register_task
import torch


# Represents a sine task
@register_task('sine')
class SineTask(FairseqTask):

    def __init__(self, args, phase, amplitude, rng):
        self.args = args
        self.datasets = {}
        self.phase = phase
        self.amplitude = amplitude
        self.rng = rng

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        phase = kwargs['phase']
        amplitude = kwargs['amplitude']
        rng = kwargs['rng']
        return cls(args=args, phase=phase, amplitude=amplitude, rng=rng)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return 1

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split not in self.datasets:
            self.datasets[split] = SineFairseqDataset(split=split, phase=self.phase, amplitude=self.amplitude, rng=self.rng,
                                                      shuffle=True, half=self.args.fp16)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        # This task doesn't really need a target dictionary, just creating a dummy dictionary for criterion abstraction
        dummy_target_dictionry = Dictionary()
        return dummy_target_dictionry

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            assert len(models) == 1
            predictions = models[0](sample['net_input']['src_tokens'])
            hypos = []
            for prediction in predictions:
                hypos.append([{'tokens': prediction,
                               'alignment': None,
                               'score': 0.0,
                               'positional_scores': torch.Tensor([])}])
            return hypos
