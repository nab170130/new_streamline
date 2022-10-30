import torch
import wilds
from torch.utils.data import Dataset


class IWildCam(Dataset):


    def __init__(self, root_dir, train, transform=None):
        self.num_tasks = 4
        self.transform = transform
        self._initialize_dataset(root_dir, train)


    def _map_locations_to_task(self, metadata_splits, metadata_info, metadata_maps):

        # Create tasks based on a balancing of the test split.
        split_indicator = metadata_maps['id_test']

        # Get split info
        keep_split_idx = []
        for idx, split in enumerate(metadata_splits):
            if split == split_indicator:
                keep_split_idx.append(idx)

        # Count how many images are in each location
        location_idx = 0
        unique_values, counts = torch.unique(metadata_info[keep_split_idx][:,location_idx], return_counts=True)
        target_size = len(keep_split_idx) / self.num_tasks
        
        # Add locations to each split, advancing to the new split once the split has the above target size by image count
        total_count = 0
        working_split_idx = 0
        location_splits = [[] for x in range(self.num_tasks)]
        for location, count in zip(unique_values, counts):
            location_splits[working_split_idx].append(location.item())
            total_count += count
            if total_count > (working_split_idx + 1) * target_size:
                working_split_idx += 1
            
        return location_splits


    def _initialize_dataset(self, root_dir, train):

        full_dataset    = wilds.get_dataset('iwildcam', root_dir=root_dir, download=True)
        metadata_splits = full_dataset._split_array
        metadata_info   = full_dataset._metadata_array
        metadata_maps   = full_dataset._split_dict

        location_splits = self._map_locations_to_task(metadata_splits, metadata_info, metadata_maps)
        location_map    = {}
        for split_idx, location_split in enumerate(location_splits):
            for location in location_split:
                location_map[location] = split_idx

        if train:
            split_indicator = metadata_maps['train']
        else:
            split_indicator = metadata_maps['id_test']

        # Get split info
        keep_split_idx = []
        for idx, split in enumerate(metadata_splits):
            if split == split_indicator:
                keep_split_idx.append(idx)

        location_idx = 0
        kept_idx_by_task = [[] for x in range(self.num_tasks)]
        for keep_idx in keep_split_idx:
            idx_location    = metadata_info[keep_idx][location_idx].item()
            if idx_location in location_map:
                mapped_task = location_map[idx_location]
                kept_idx_by_task[mapped_task].append(keep_idx)

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

        return image, label


    def __len__(self):
        return sum([len(task) for task in self.task_idx_partitions])   