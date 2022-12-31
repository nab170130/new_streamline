import time
from mmcv import Config

from torchvision import transforms

import numpy as np
import torch

from .bdd100k import BDD100K
from .fmow import FMOW
from .iwildcam import IWildCam
from .kitti_fog import KITTIFog
from .cityscapes_rain import CityscapesRain
from .office_31 import Office31
from .organ_mnist import OrganMNIST
from .perm_mnist import PermutedMNIST
from .rot_mnist import RotatedMNIST

class DatasetFactory:
    
    def __init__(self, root):
        
        self.root_directory = root
    
    
    def get_dataset(self, dataset_name):
        
        if dataset_name == "RotatedMNIST":

            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            full_train_dataset = RotatedMNIST(self.root_directory, train=True, num_tasks=5)
            nclasses = 8

        elif dataset_name == "OrganMNIST":

            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            full_train_dataset = OrganMNIST(self.root_directory, train=True)
            nclasses = 11

        elif dataset_name == "PermutedMNIST":

            test_transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
            full_train_dataset = PermutedMNIST(self.root_directory, train=True, num_tasks=5)
            nclasses = 10

        elif dataset_name == "Office31":

            test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            full_train_dataset = Office31(self.root_directory, train=True)
            nclasses = 31

        elif dataset_name == "IWildCam":

            test_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            full_train_dataset = IWildCam(self.root_directory, train=True)
            nclasses = 182

        elif dataset_name == "FMOW":

            test_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            full_train_dataset = FMOW(self.root_directory, train=True)
            nclasses = 62

        elif dataset_name == "BDD100K":

            # To get the test transform, we simply load the config and get its test pipeline
            bdd100k_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_bdd100k_cocofmt.py"
            bdd100k_config      = Config.fromfile(bdd100k_config_path)

            # Build the dataset using the train configuration
            full_train_dataset  = BDD100K(self.root_directory, train=True)
            test_transform      = bdd100k_config.test_pipeline
            nclasses            = len(bdd100k_config.CLASSES)

        elif dataset_name == "KITTIFog":

            # To get the test transform, we simply load the config and get its test pipeline
            kitti_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_kitti_cocofmt.py"
            kitti_config      = Config.fromfile(kitti_config_path)

            # Build the dataset using the train configuration
            full_train_dataset  = KITTIFog(self.root_directory, train=True)
            test_transform      = kitti_config.test_pipeline
            nclasses            = len(kitti_config.CLASSES)

        elif dataset_name == "CityscapesRain":

            # To get the test transform, we simply load the config and get its test pipeline
            cityscapes_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_cityscapes_cocofmt.py"
            cityscapes_config      = Config.fromfile(cityscapes_config_path)

            # Build the dataset using the train configuration
            full_train_dataset  = CityscapesRain(self.root_directory, train=True)
            test_transform      = cityscapes_config.test_pipeline
            nclasses            = len(cityscapes_config.CLASSES)

        else:

            raise ValueError("Dataset not implemented!")

        return full_train_dataset, test_transform, nclasses
    
    
    def get_initial_split(self, dataset_name, per_task_initial_size = 100):
        
        if dataset_name == "RotatedMNIST":

            redundancy_factor = 1
            full_train_dataset = RotatedMNIST(self.root_directory, train=True, num_tasks=5)

        elif dataset_name == "OrganMNIST":

            redundancy_factor = 1
            full_train_dataset = OrganMNIST(self.root_directory, train=True)

        elif dataset_name == "PermutedMNIST":

            redundancy_factor = 1
            full_train_dataset = PermutedMNIST(self.root_directory, train=True, num_tasks=5)

        elif dataset_name == "Office31":

            redundancy_factor = 1
            full_train_dataset = Office31(self.root_directory, train=True)

        elif dataset_name == "IWildCam":

            redundancy_factor = 1
            full_train_dataset = IWildCam(self.root_directory, train=True)

        elif dataset_name == "FMOW":

            redundancy_factor = 1
            full_train_dataset = FMOW(self.root_directory, train=True)

        elif dataset_name == "BDD100K":

            # SET TO MATCH IN (un)lim_memory_det_experiment.py -- SAME FOR ALL OBJ DET.
            redundancy_factor = 2
            full_train_dataset  = BDD100K(self.root_directory, train=True)

        elif dataset_name == "KITTIFog":

            # SET TO MATCH IN (un)lim_memory_det_experiment.py -- SAME FOR ALL OBJ DET.
            redundancy_factor = 2
            full_train_dataset  = KITTIFog(self.root_directory, train=True)

        elif dataset_name == "CityscapesRain":

            # SET TO MATCH IN (un)lim_memory_det_experiment.py -- SAME FOR ALL OBJ DET.
            redundancy_factor = 2
            full_train_dataset  = CityscapesRain(self.root_directory, train=True)

        else:

            raise ValueError("Dataset not implemented!")

        # Seed the RNG for generating initial splits so that all runs share the same initial starting point.
        torch.manual_seed(40)
        np.random.seed(40)

        initial_training_split_idx_partitions = []
        initial_unlabeled_split_idx_partitions = []
        for task_num, task_idx_partition in enumerate(full_train_dataset.task_idx_partitions):
            task_sample_size = per_task_initial_size // redundancy_factor
            if task_num == len(full_train_dataset.task_idx_partitions) - 1:
                task_sample_size = task_sample_size // 5
            training_task_idx_partition = np.random.choice(task_idx_partition, size=task_sample_size, replace=False).tolist()
            training_task_idx_partition = training_task_idx_partition * redundancy_factor
            unlabeled_task_idx_partition = list(set(task_idx_partition) - set(training_task_idx_partition))
            initial_training_split_idx_partitions.append(training_task_idx_partition)
            initial_unlabeled_split_idx_partitions.append(unlabeled_task_idx_partition)
        
        # Change seed after sampling using current unix time. Modulo to be within 2**32 - 1.
        new_seed = time.time_ns() % 1000000
        torch.manual_seed(new_seed)
        np.random.seed(new_seed)

        return initial_training_split_idx_partitions, initial_unlabeled_split_idx_partitions


    def get_eval_dataset(self, dataset_name):

        if dataset_name == "RotatedMNIST":

            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            test_dataset = RotatedMNIST(self.root_directory, train=False, num_tasks=5, transform=test_transform)
            nclasses = 8

        elif dataset_name == "OrganMNIST":

            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            test_dataset = OrganMNIST(self.root_directory, train=True, transform=test_transform)
            nclasses = 11

        elif dataset_name == "PermutedMNIST":

            test_transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
            test_dataset = PermutedMNIST(self.root_directory, train=False, num_tasks=5, transform=test_transform)
            nclasses = 10

        elif dataset_name == "Office31":

            test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            test_dataset = Office31(self.root_directory, train=False, transform=test_transform)
            nclasses = 31

        elif dataset_name == "IWildCam":

            test_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            test_dataset = IWildCam(self.root_directory, train=False, transform=test_transform)
            nclasses = 182

        elif dataset_name == "FMOW":

            test_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            test_dataset = FMOW(self.root_directory, train=False, transform=test_transform)
            nclasses = 62

        elif dataset_name == "BDD100K":

            # To get the test transform, we simply load the config and get its test pipeline
            bdd100k_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_bdd100k_cocofmt.py"
            bdd100k_config      = Config.fromfile(bdd100k_config_path)

            # Build the dataset using the train configuration
            test_transform  = bdd100k_config.test_pipeline
            test_dataset    = BDD100K(self.root_directory, train=False)
            nclasses        = len(bdd100k_config.CLASSES)

        elif dataset_name == "KITTIFog":

            # To get the test transform, we simply load the config and get its test pipeline
            kitti_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_kitti_cocofmt.py"
            kitti_config      = Config.fromfile(kitti_config_path)

            # Build the dataset using the train configuration
            test_transform  = kitti_config.test_pipeline
            test_dataset    = KITTIFog(self.root_directory, train=False)
            nclasses        = len(kitti_config.CLASSES)
        
        elif dataset_name == "CityscapesRain":

            # To get the test transform, we simply load the config and get its test pipeline
            cityscapes_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_kitti_cocofmt.py"
            cityscapes_config      = Config.fromfile(cityscapes_config_path)

            # Build the dataset using the train configuration
            test_transform  = cityscapes_config.test_pipeline
            test_dataset    = CityscapesRain(self.root_directory, train=False)
            nclasses        = len(cityscapes_config.CLASSES)

        else:

            raise ValueError(F"Dataset {dataset_name} not implemented!")

        return test_dataset, test_transform, nclasses