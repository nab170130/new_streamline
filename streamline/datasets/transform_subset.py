from torch.utils.data import Dataset

class TransformSubset(Dataset):
    
    def __init__(self, base_dataset, task_idx_partitions, is_labeled=True, transform=None):
        
        self.base_dataset = base_dataset
        self.task_idx_partitions = task_idx_partitions
        self.is_labeled = is_labeled
        self.transform = transform
        

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
        data, label = self.base_dataset[mapped_index]
        
        if self.transform is not None:
            data = self.transform(data)
            
        if self.is_labeled:
            return data, label
        else:
            return data
        

    def __len__(self):
        return sum([len(task) for task in self.task_idx_partitions]) 