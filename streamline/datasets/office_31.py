import time
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

import os
import PIL.Image as Image


class Office31(Dataset):

    def __init__(self, root_dir, train, transform=None):
        self.num_tasks = 3
        self._initialize_dataset(os.path.join(root_dir, "office31"), train)
        self.transform = transform


    def _initialize_dataset(self, root_dir, train):

        class Office31Subset(Dataset):
        
            def __init__(self, root, file_location_split, transform=None):
                
                self.root = root
                self.file_location_split = file_location_split
                self.transform = transform
                
                self.class_name_to_idx_map = self._get_class_idx_map()
                
            def _get_class_idx_map(self):
                
                # Get all directories in one of the domain folders
                amazon_folder = os.path.join(self.root, "amazon", "images")
                all_classes = sorted(os.listdir(amazon_folder)) 
                
                # Create a dictionary that maps class name to a number
                name_to_idx_map = {}
                for num, class_name in enumerate(all_classes):
                    name_to_idx_map[class_name] = num
                    
                return name_to_idx_map
                
            def __getitem__(self, index):
                
                # Get image location
                rel_path = self.file_location_split[index]
                abs_path = os.path.join(self.root, rel_path)
                
                # Get the class of the image, which is obtained from 
                # .../class_name/frame_num.jpg
                class_name = rel_path.split("/")[-2]
                image_class = self.class_name_to_idx_map[class_name]
        
                with open(abs_path, "rb") as f:
                    image = Image.open(f).convert("RGB")
                    if self.transform is not None:
                        image = self.transform(image)
                
                return image, image_class        
                
            def __len__(self):
                return len(self.file_location_split)
            
        def get_split_list(path_to_file):
            with open(path_to_file, "r") as split_file:
                list_of_paths = [line.rstrip() for line in split_file]
            return list_of_paths

        split = "train" if train else "test"

        # Get all splits
        amazon_split_file_loc = os.path.join(root_dir, "office_splits", F"amazon_{split}.txt")
        dslr_split_file_loc = os.path.join(root_dir, "office_splits", F"dslr_{split}.txt")
        webcam_split_file_loc = os.path.join(root_dir, "office_splits", F"webcam_{split}.txt")

        amazon_split = get_split_list(amazon_split_file_loc)
        dslr_split = get_split_list(dslr_split_file_loc)
        webcam_split = get_split_list(webcam_split_file_loc)
        
        # Instantiate each task
        amazon_task = Office31Subset(root_dir, amazon_split, transform=None)
        webcam_task = Office31Subset(root_dir, webcam_split, transform=None)
        dslr_task = Office31Subset(root_dir, dslr_split, transform=None)
        
        # If this is the test set, we need to ensure balance across tasks.
        if not train:
            torch.manual_seed(40)
            np.random.seed(40)

            # Choose enough instances from each task and assign subsets
            min_task_test_size = min([len(amazon_task), len(webcam_task), len(dslr_task)])
            selected_amazon_idx = np.random.choice(len(amazon_task), size=min_task_test_size, replace=False).tolist()
            selected_webcam_idx = np.random.choice(len(webcam_task), size=min_task_test_size, replace=False).tolist()
            selected_dslr_idx = np.random.choice(len(dslr_task), size=min_task_test_size, replace=False).tolist()
            amazon_task = Subset(amazon_task, selected_amazon_idx)
            webcam_task = Subset(webcam_task, selected_webcam_idx)
            dslr_task = Subset(dslr_task, selected_dslr_idx)

            # Change seed after sampling using current unix time. Modulo to be within 2**32 - 1.
            new_seed = time.time_ns() % 1000000
            torch.manual_seed(new_seed)
            np.random.seed(new_seed)

        task_list = [amazon_task, dslr_task, webcam_task]
        
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