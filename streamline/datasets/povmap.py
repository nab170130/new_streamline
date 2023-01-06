import torch
import wilds
from torch.utils.data import Dataset


class PovertyMap(Dataset):


    def __init__(self, root_dir, train, transform=None):
        self.num_tasks = 2
        self.transform = transform
        self._initialize_dataset(root_dir, train)


    def _initialize_dataset(self, root_dir, train):

        full_dataset    = wilds.get_dataset('poverty', root_dir=root_dir, download=True)
        
        metadata_splits = full_dataset._split_array
        metadata_info   = full_dataset._metadata_array
        metadata_maps   = full_dataset._split_dict
        
        # Keep only those indices belonging to the specified split. We use only in-distribution examples when evaluating.
        if train:
            split_indicator = metadata_maps['train']
        else:
            split_indicator = metadata_maps['id_test']

        urban_attr_idx = -1
        for attr_idx, attr_name in enumerate(full_dataset._metadata_fields):
            if attr_name == 'urban':
                urban_attr_idx = attr_idx
                break            

        kept_idx_by_task = [[] for x in range(self.num_tasks)]
        for idx, split in enumerate(metadata_splits):
            if split == split_indicator:
                task_number = int(metadata_info[idx][urban_attr_idx])
                kept_idx_by_task[task_number].append(idx)

        to_base_idx_mapping = []
        task_idx_partitions = []
        start_idx           = 0
        for task_base_mapping in kept_idx_by_task:
            to_base_idx_mapping.extend(task_base_mapping)
            task_idx_partitions.append(list(range(start_idx, start_idx + len(task_base_mapping))))
            start_idx = start_idx + len(task_base_mapping)

        self.base_dataset           = full_dataset
        self.to_base_idx_mapping    = to_base_idx_mapping
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
        base_mapped_idx = self.to_base_idx_mapping[index]
        image, label, _ = self.base_dataset[base_mapped_idx]
        if self.transform is not None:
            image = self.transform(image)

        # Convert label to binary classification: 0 if negative index score, 1 if positive.
        label = 1 if label > 0 else 0

        return image, label


    def __len__(self):
        return sum([len(task) for task in self.task_idx_partitions])   