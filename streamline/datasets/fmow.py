import numpy as np
import time
import torch
import wilds
from torch.utils.data import Dataset


class FMOW(Dataset):

    def __init__(self, root_dir, train, transform=None):
        self.num_tasks = 3
        self.transform = transform
        self._initialize_dataset(root_dir, train)


    def _initialize_dataset(self, root_dir, train):

        full_fmow_dataset               = wilds.get_dataset("fmow", download=True, root_dir=root_dir)
        full_fmow_split_dataset         = full_fmow_dataset.get_subset("train" if train else "test")
        region_name_to_region_code_map  = {0: "asia", 1: "europe", 2: "africa", 3: "americas", 4: "oceania", 5: "other"}

        # Partition the chosen split into geographical region.
        task_split_idx = [[] for x in range(self.num_tasks)]
        metadata = full_fmow_split_dataset.metadata_array
        for full_fmow_split_idx, (region, _, _, _) in enumerate(metadata):      
            if region in list(range(self.num_tasks)):
                task_split_idx[region].append(full_fmow_split_idx)

        # If this is the test set, ensure that there is an equal balance across tasks.
        if not train:
            torch.manual_seed(40)
            np.random.seed(40)

            # Choose enough instances from each task and assign subsets
            min_task_test_size = min([len(x) for x in task_split_idx])
            subset_task_split_idx = []
            for superset_task_split_idx in task_split_idx:
                selected_subset_idx = np.random.choice(superset_task_split_idx, size=min_task_test_size, replace=False).tolist()
                subset_task_split_idx.append(selected_subset_idx)

            task_split_idx = subset_task_split_idx

            # Change seed after sampling using current unix time. Modulo to be within 2**32 - 1.
            new_seed = time.time_ns() % 1000000
            torch.manual_seed(new_seed)
            np.random.seed(new_seed)

        task_idx_partitions = []
        base_mapping = []
        start_idx = 0
        for task_split in task_split_idx:
            task_idx_partitions.append(list(range(start_idx, start_idx + len(task_split))))
            start_idx = start_idx + len(task_split)
            base_mapping.extend(task_split)

        self.base_dataset           = full_fmow_split_dataset
        self.base_mapping           = base_mapping
        self.task_idx_partitions    = task_idx_partitions


    def get_task_number_and_index_in_task(self, index):

        # Determine to which partition this index belongs and the particular index within that task.
        working_idx = index
        for task_num, task in enumerate(self.task_idx_partitions):
            if working_idx - len(task) < 0:
                break
            working_idx = working_idx - len(task)
            
        return task_num, working_idx


    def __getitem__(self, index):
        
        # Retrieve the image and the label
        base_idx                = self.base_mapping[index]
        image, label, _         = self.base_dataset[base_idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return sum([len(task) for task in self.task_idx_partitions])   