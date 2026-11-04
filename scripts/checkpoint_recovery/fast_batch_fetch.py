"""Fetch identical sequential TensorDataset batches without per-row collation.

The original DataLoader iterator, batch sampler and CPU RNG draws are retained.
Only CPU batch assembly changes; tensors are made contiguous exactly as torch.stack.
"""
import torch
from torch.utils.data import SequentialSampler, TensorDataset, default_collate


def install_contiguous_batch_fetch(loader):
    if (type(loader.dataset) is not TensorDataset or loader.num_workers != 0
            or type(loader.sampler) is not SequentialSampler or loader.drop_last
            or loader.batch_size is None or loader.collate_fn is not default_collate):
        raise ValueError('Fast batch fetch requires the original sequential TensorDataset loader')
    tensors = loader.dataset.tensors
    checked = False

    def get_batch(indices):
        nonlocal checked
        start, stop = indices[0], indices[-1] + 1
        if indices != list(range(start, stop)):
            raise ValueError('Non-sequential batch indices')
        batch = tuple(tensor[start:stop].contiguous() for tensor in tensors)
        if not checked:
            reference = default_collate([loader.dataset[i] for i in indices])
            if any(not torch.equal(actual, expected) or actual.stride() != expected.stride()
                   for actual, expected in zip(batch, reference)):
                raise ValueError('Vectorized batch differs from original collation')
            checked = True
            print(f'BATCH_FETCH exact_original_batch=True rows={len(indices)}', flush=True)
        return batch

    loader.dataset.__getitems__ = get_batch
    loader.collate_fn = lambda batch: batch
