from torch.utils.data import Dataset, Subset

import medmnist
import numpy as np
import os
import torch
import time


class OrganMNIST(Dataset):

    def __init__(self, root_dir, train, transform=None):
        self.num_tasks = 3
        self.transform = transform

        self._initialize_dataset(os.path.join(root_dir, "medmnist"), train)


    def _initialize_dataset(self, base_directory, train):

        split = "train" if train else "test"

        # Initialize the task list, which is simply the concatenation of OrganAMNIST,
        # OrganCMNIST, and OrganSMNIST.
        label_transform = lambda x : x[0]
        organ_amnist = medmnist.OrganAMNIST(split, root=base_directory, download=True, target_transform=label_transform)
        organ_cmnist = medmnist.OrganCMNIST(split, root=base_directory, download=True, target_transform=label_transform)
        organ_smnist = medmnist.OrganSMNIST(split, root=base_directory, download=True, target_transform=label_transform)
        task_list = [organ_amnist, organ_cmnist, organ_smnist]
        
        # If this is the test set, we need to ensure balance across tasks.
        if not train:
            torch.manual_seed(40)
            np.random.seed(40)

            # Choose enough instances from each task and assign subsets
            min_task_test_size = min([len(organ_amnist), len(organ_cmnist), len(organ_smnist)])
            selected_amnist_idx = np.random.choice(len(organ_amnist), size=min_task_test_size, replace=False).tolist()
            selected_cmnist_idx = np.random.choice(len(organ_cmnist), size=min_task_test_size, replace=False).tolist()
            selected_smnist_idx = np.random.choice(len(organ_smnist), size=min_task_test_size, replace=False).tolist()
            organ_amnist = Subset(organ_amnist, selected_amnist_idx)
            organ_cmnist = Subset(organ_cmnist, selected_cmnist_idx)
            organ_smnist = Subset(organ_smnist, selected_smnist_idx)

            # Change seed after sampling using current unix time. Modulo to be within 2**32 - 1.
            new_seed = time.time_ns() % 1000000
            torch.manual_seed(new_seed)
            np.random.seed(new_seed)

        # Create the task index partition field now that the size of each task is known
        task_idx_partitions = []
        start_idx = 0
        for task in task_list:
            end_idx = start_idx + len(task)
            task_idx_partition = list(range(start_idx, end_idx))
            task_idx_partitions.append(task_idx_partition)
            start_idx = end_idx

        self.task_idx_partitions = task_idx_partitions
        self.task_list = task_list


    def get_task_number_and_index_in_task(self, index):

        # Determine to which partition this index belongs and the particular index within that task.
        working_idx = index
        for task_num, task in enumerate(self.task_list):
            if working_idx - len(task) < 0:
                break
            working_idx = working_idx - len(task)
            
        return task_num, working_idx


    def __getitem__(self, index):
        
        # Determine to which partition this index belongs and the particular index within that task.
        task_num, in_task_idx = self.get_task_number_and_index_in_task(index)

        # Retrieve the image and the label
        image, label = self.task_list[task_num][in_task_idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return sum([len(task) for task in self.task_idx_partitions])   