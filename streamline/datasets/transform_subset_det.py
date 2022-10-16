from torch.utils.data import Dataset

import numpy as np

class TransformSubsetDetection(Dataset):
    
    def __init__(self, base_dataset, task_idx_partitions):
        
        self.base_dataset = base_dataset
        self.task_idx_partitions = task_idx_partitions
        self.CLASSES = self.base_dataset.CLASSES
        self._set_group_flag()


    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            task_num, in_task_idx   = self.get_task_number_and_index_in_task(i)
            base_idx                = self.task_idx_partitions[task_num][in_task_idx]
            self.flag[i] = self.base_dataset.flag[base_idx]


    def get_task_number_and_index_in_task(self, index):

        # Determine to which partition this index belongs and the particular index within that task.
        working_idx = index
        for task_num, task in enumerate(self.task_idx_partitions):
            if working_idx - len(task) < 0:
                break
            working_idx = working_idx - len(task)
            
        return task_num, working_idx


    def __getitem__(self, index):
        
        # Determine to which partition this index belongs and the particular index within that task.
        task_num, working_idx = self.get_task_number_and_index_in_task(index)

        # Map the index back to the full labeled dataset and use it to retrieve the data
        mapped_index = self.task_idx_partitions[task_num][working_idx]
        instance = self.base_dataset[mapped_index]
            
        return instance
        

    def __len__(self):
        return sum([len(task) for task in self.task_idx_partitions]) 