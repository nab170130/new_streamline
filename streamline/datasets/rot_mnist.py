from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, Subset


class RotatedMNIST(Dataset):

    def __init__(self, root_dir, train, num_tasks, transform=None):
        self.num_tasks = num_tasks
        self._initialize_dataset(root_dir, train)
        self.transform = transform


    def _initialize_dataset(self, root_dir, train):

        def _remap_rotated_mnist_label(old_label):

            # Use the following label map to transform the old label to the new label
            label_map = {0:0,
                        1:1,
                        2:2,
                        3:3,
                        4:4,
                        5:5,
                        7:6,
                        8:7}

            return label_map[old_label]


        # Only train/test on classes [0,1,2,3,4,5,7,8] since 6/9 are not label-preserving under rotation
        mnist_dataset = MNIST(root_dir, train=train, download=True)
        keep_idx = []

        for index, (_, label) in enumerate(mnist_dataset):
            if label not in [6,9]:
                keep_idx.append(index)

        # Create the task index partition field now that the size of each task is known
        task_idx_partitions = []
        start_idx = 0
        for i in range(self.num_tasks):
            end_idx = start_idx + len(keep_idx)
            task_idx_partition = list(range(start_idx, end_idx))
            task_idx_partitions.append(task_idx_partition)
            start_idx = end_idx

        # Rotate the MNIST images in each task.
        rot_per_task = 360 / self.num_tasks

        task_list = []

        # Map each label to a format that torch uses. Furthermore, rotate images in a task by a set angle.
        for i in range(self.num_tasks):

            task_angle = rot_per_task * i
            task_i_rot_transform = transforms.Compose([transforms.RandomRotation([task_angle, task_angle])])
            mnist_dataset = MNIST(root_dir, train=train, download=True, transform = task_i_rot_transform, target_transform = _remap_rotated_mnist_label)
            task_subset = Subset(mnist_dataset, keep_idx)     
            task_list.append(task_subset)

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

        # Retrieve the image and the label. Both have already gone under the necessary transformation.
        image, label = self.task_list[task_num][within_task_idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return sum([len(task) for task in self.task_idx_partitions])    