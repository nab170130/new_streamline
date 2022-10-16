from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset

import numpy as np


class PermutedMNIST(Dataset):

    def __init__(self, root_dir, train, num_tasks, transform=None):
        self.num_tasks = num_tasks
        self._initialize_dataset(root_dir, train)
        self.transform = transform


    def _initialize_dataset(self, root_dir, train):

        # In permuted MNIST, each task is formed by applying a permutation on each of MNIST's images.
        # Different tasks use different permutations. Here, a list of permutations is generated.
        class ImagePermutation:
            
            def __init__(self, image_dim):
                
                if len(image_dim) != 3:
                    raise ValueError("image_dim arg must be length 3!")
                
                self.image_dim = image_dim
                
                # Generate a random permutation.
                num_pixels_to_shuffle = image_dim[0] * image_dim[1] * image_dim[2]
                self.permutation = np.arange(num_pixels_to_shuffle)
                np.random.shuffle(self.permutation)
                
            def __call__(self, image):
                
                # Image should be a tensor. Flatten it, shuffle it, and reshape it back.
                flattened_image = image.flatten()
                permuted_flattened_image = flattened_image[self.permutation]
                permuted_image = permuted_flattened_image.reshape(self.image_dim)
                return permuted_image
            
        # First, get the image dimension of MNIST
        mnist_dataset = MNIST(root_dir, train=train, download=True, transform=transforms.ToTensor())
        image_dim = mnist_dataset[0][0].shape
        
        # Create a list of permutation functions. Seed the RNG to have reproducible tasks.
        np.random.seed(42)
        permutations = [ImagePermutation(image_dim) for x in range(self.num_tasks)]
        
        # Create the task index partition field now that the size of each task is known
        task_idx_partitions = []
        start_idx = 0
        for i in range(self.num_tasks):
            end_idx = start_idx + len(mnist_dataset)
            task_idx_partition = list(range(start_idx, end_idx))
            task_idx_partitions.append(task_idx_partition)
            start_idx = end_idx

        # Create a transform unique to each task that permutes the images
        task_list = []
        for i in range(self.num_tasks):
            task_i_perm_transform = transforms.Compose([transforms.ToTensor(), permutations[i]])
            mnist_dataset = MNIST(root_dir, train=train, download=True, transform = task_i_perm_transform)
            task_list.append(mnist_dataset)

        self.task_idx_partitions = task_idx_partitions
        self.task_list = task_list


    def get_task_number_and_index_in_task(self, index):

        # Determine to which partition this index belongs and the particular index within that task.
        task_size = len(self.task_list[0])
        task_num = index // task_size
        within_task_idx = index % task_size
            
        return task_num, within_task_idx


    def __getitem__(self, index):
        
        # Determine to which partition this index belongs and the particular index within that task.
        task_num, within_task_idx = self.get_task_number_and_index_in_task(index)

        # Retrieve the image and the label.
        image, label = self.task_list[task_num][within_task_idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return sum([len(task) for task in self.task_idx_partitions])    