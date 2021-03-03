import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from fairseq.data.fairseq_dataset import FairseqDataset


def sine(phase, amplitude, x):
    return np.sin(x + phase) * amplitude


class SineDataset(Dataset):
    def __init__(self, split, phase, amplitude, rng):
        self.rng = rng
        if split == 'train':
            # All of the x points
            self.x = np.linspace(-5, 5, 50, dtype=np.float32)[:, None]
            self.y = sine(phase=phase, amplitude=amplitude, x=self.x)
        elif split == 'test':
            # All of the x points
            all_data = np.linspace(-5, 5, 50, dtype=np.float32)[:, None]
            self.x = all_data[np.array([3, 3, 39, 9, 19, 21, 36, 23, 6, 24])]
            self.y = sine(phase=phase, amplitude=amplitude, x=self.x)
        elif split == 'valid':
            # Create a validation split
            self.x = np.linspace(-4, 4, 50, dtype=np.float32)[:, None]
            self.y = sine(phase=phase, amplitude=amplitude, x=self.x)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class SineFairseqDataset(SineDataset, FairseqDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self, split, phase, amplitude, rng, shuffle, half):
        super().__init__(split=split, phase=phase, amplitude=amplitude, rng=rng)
        self.shuffle = shuffle
        self.half = half

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        mini_batch = default_collate(samples)
        assert len(mini_batch) == 2
        if self.half:
            mini_batch[0] = mini_batch[0].half()
            mini_batch[1] = mini_batch[1].half()
        id = torch.LongTensor(range(len(samples)))
        nsentences = len(samples)
        lengths = torch.ones(nsentences, 1)
        mini_batch_dict = {'net_input': {'src_tokens': mini_batch[0], 'src_lengths': lengths},
                           'target': mini_batch[1],
                           'id': id,
                           'nsentences': nsentences}
        return mini_batch_dict

    def get_dummy_batch(self, num_tokens, max_positions, src_len=1, tgt_len=1):
        """Return a dummy batch with a given number of tokens."""
        x = torch.zeros(num_tokens, 1)
        y = torch.zeros(num_tokens, 1)
        l = torch.ones(num_tokens, 1)
        id = torch.LongTensor(range(num_tokens))
        return {'net_input': {'src_tokens': x, 'src_lengths': l}, 'target': y, 'id': id, 'nsentences': num_tokens}

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return 1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return 1

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        num_train_examples = len(self)
        if self.shuffle:
            return self.rng.permutation(num_train_examples)
        else:
            return range(num_train_examples)

