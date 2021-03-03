import numpy as np

from fairseq.data.fairseq_dataset import FairseqDataset


class MetaFairseqDataset(FairseqDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self, split, meta_tasks):
        self.split = split
        self.meta_tasks = meta_tasks

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return samples

    def get_dummy_batch(self, num_tokens, max_positions):
        """Return a dummy batch with a given number of tokens."""
        samples = self.meta_tasks[:num_tokens]
        return samples

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return 1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when filtering a dataset with
        ``--max-positions``."""
        return 1

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        # TODO: Add Curriculum
        num_tasks = len(self.meta_tasks)
        return np.arange(num_tasks)

    def __getitem__(self, index):
        return self.meta_tasks[index]
