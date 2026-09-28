"""Read explicitly named physical joint channels unused by the training adapter."""
import numpy as np
import pyarrow.parquet as pq
import torch


def read_joints(dataset, episode, indices, entries):
    path = dataset._local_data_path(dataset._data_rel_path(episode))
    source = pq.ParquetFile(path)
    keys = {entry['key'] for entry in entries}
    columns = keys | ({'episode_index', 'index'} & set(source.schema_arrow.names))
    table = dataset._slice_episode_table(source.read(columns=sorted(columns)), episode)
    values = np.zeros((len(indices), dataset.canonical_dim), dtype=np.float32)
    pad = torch.ones(dataset.canonical_dim, dtype=torch.bool)
    for entry in entries:
        key = entry['key']
        names = dataset.info['features'][key].get('names')
        requested = entry['names']
        if not isinstance(names, list) or any(names.count(name) != 1 for name in requested):
            raise ValueError(f'{key}: requested joint names must occur exactly once in source metadata')
        low, high = map(int, entry['target_slice'])
        if not (0 <= low < high <= values.shape[1]) or high - low != len(requested):
            raise ValueError(f'Invalid joint target slice: {entry}')
        if not bool(pad[low:high].all()):
            raise ValueError(f'Overlapping joint target slice: {entry}')
        raw = np.asarray(table[key].to_pylist(), dtype=np.float32)
        values[:, low:high] = raw[np.asarray(indices)][:, [names.index(name) for name in requested]]
        pad[low:high] = False
    return values, pad
