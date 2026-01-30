

import argparse
import os

from datasets import Dataset, load_dataset
from verl.utils.hdfs_io import copy, makedirs






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/hh')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=7500)
    parser.add_argument('--test_size', type=int, default=5000)

    args = parser.parse_args()
    data_source = 'hh'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset('trl-lib/ultrafeedback_binarized', split='train')


    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    raw_dataset = raw_dataset.shuffle(seed=42)
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example['chosen'][0]['content']
            solution = example['chosen'][1]['content']
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "alignment",
                "reward_model": {
                    "style": "model",
                    "ground_truth": {
                        "prompt": question,
                    },
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
