import pickle
from copy import deepcopy
import os
import numpy as np


class IndexedDataset:
    def __init__(self, path, prefix, num_cache=1, segment_size=1024):
        super().__init__()
        self.path = path
        self.segment_size = segment_size
        segment_count = len([f for f in os.listdir(path) if f.startswith(prefix) and f.endswith('.idx')])
        self.data_offsets = [
            np.load(os.path.join(path, f"{prefix}_{segment_idx}.idx"), allow_pickle=True).item()['offsets']
            for segment_idx in range(segment_count)
        ]
        self.data_files = [
            open(os.path.join(path, f"{prefix}_{segment_idx}.data"), 'rb', buffering=-1)
            for segment_idx in range(segment_count)
        ]
        self.total_size = sum(len(offsets) - 1 for offsets in self.data_offsets)
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= self.total_size:
            raise IndexError('index out of range')

    def __del__(self):
        for f in self.data_files:
            f.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        segment_idx = i // self.segment_size
        idx = i % self.segment_size
        self.data_files[segment_idx].seek(self.data_offsets[segment_idx][idx])
        b = self.data_files[segment_idx].read(self.data_offsets[segment_idx][idx + 1] - self.data_offsets[segment_idx][idx])
        item = pickle.loads(b) # 反序列化
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.data_offsets) - 1

class IndexedDatasetBuilder:
    def __init__(self, path, prefix, segment_size=1024):
        self.segment_idx = 0
        self.segment_size = segment_size
        self.segment_item_count = 0
        self.path = path
        self.prefix = prefix
        self.out_file = open(os.path.join(path, f"{prefix}_{self.segment_idx}.data"), 'wb')
        self.byte_offsets = [0]

    def add_item(self, item):
        s = pickle.dumps(item)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)
        self.segment_item_count += 1
        if self.segment_item_count >= self.segment_size:
            self.finalize()
            self.segment_idx += 1
            self.segment_item_count = 0
            self.out_file = open(os.path.join(self.path, f"{self.prefix}_{self.segment_idx}.data"), 'wb')
            self.byte_offsets = [0]

    def finalize(self):
        self.out_file.close()
        with open(os.path.join(self.path, f"{self.prefix}_{self.segment_idx}.idx"), 'wb') as f:
            np.save(f, {'offsets': self.byte_offsets})


if __name__ == "__main__":
    import random
    from tqdm import tqdm
    import os
    ds_path = '/tmp/indexed_ds_example'
    size = 100
    items = [{"a": np.random.normal(size=[10000, 10]),
              "b": np.random.normal(size=[10000, 10])} for i in range(size)]
    builder = IndexedDatasetBuilder(ds_path)
    for i in tqdm(range(size)):
        builder.add_item(items[i])
    builder.finalize()
    ds = IndexedDataset(ds_path)
    for i in tqdm(range(10000)):
        idx = random.randint(0, size - 1)
        assert (ds[idx]['a'] == items[idx]['a']).all()
