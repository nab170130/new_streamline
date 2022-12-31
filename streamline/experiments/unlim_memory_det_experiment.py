import copy
from .experiment import Experiment
from ..al_strategies import ALFactory
from ..datasets import DatasetFactory, TransformSubsetDetection
from ..metrics import MetricFactory
from ..models import ModelFactory
from ..persistence import add_al_round, add_metric_time_value_to_run, format_augmentation, generate_save_locations_for_al_round_det, get_absolute_paths_det, get_most_recent_round_info, atomic_torch_save, atomic_json_save, get_all_rounds_from_run
from ..training_loops import TrainingLoopFactory
from ..utils import sample_sequential_access_chain, sample_rare_access_chain, sample_random_access_chain

from mmcv import Config
from mmcv.runner import load_checkpoint

import json
import numpy as np
import os
import torch

class UnlimitedMemoryDetectionExperiment(Experiment):
    

    def __init__(self, comm_lock, comm_sem, parent_pipe, gpu_name, base_dataset_directory, base_exp_directory, db_loc, obj_det_config_path):

        super(UnlimitedMemoryDetectionExperiment, self).__init__(comm_lock, comm_sem, parent_pipe, gpu_name, base_dataset_directory, base_exp_directory, db_loc)
        self.round_number = 0
        self.obj_det_config_path = obj_det_config_path


    def experiment(self, train_dataset_name, model_architecture_name, limited_memory, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size, num_rounds, delete_large_objects_after_run=False):

        # Do each AL round.
        self.num_rounds = num_rounds
        while self.round_number < num_rounds:
            self.do_al_round(train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size, num_rounds)
            
        if delete_large_objects_after_run:

            # If the large objects are going to be deleted, then calculate each metric before deletion.
            round_info_list = get_all_rounds_from_run(self.db_loc, train_dataset_name, model_architecture_name, 0, arrival_pattern, run_number,
                                                        training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size)

            # Get the number of tasks for the dataset and all the metrics to compute.
            num_tasks = len(DatasetFactory(self.base_dataset_directory).get_dataset(train_dataset_name)[0].task_idx_partitions)
            all_metrics_to_compute = ["mAP", "labeled_instances", "task_identification_accuracy"]
            for task in range(num_tasks):
                all_metrics_to_compute.append(F"task_presence_{task}")
                all_metrics_to_compute.append(F"per_task_mAP_{task}")
            
            for round_id, current_epoch, dataset_split_path, model_weight_path, opt_state_path, lr_state_path, completed_unix_time in round_info_list:
                
                # Don't do anything if there are no large objects. This means that the metrics should have already been computed.
                head, tail = os.path.split(model_weight_path)
                abs_split_path, abs_weight_folder = get_absolute_paths_det(self.base_exp_directory, dataset_split_path, head)
                abs_weight_path = os.path.join(abs_weight_folder, tail)
                if not os.path.exists(abs_split_path):
                    continue
                if not os.path.exists(abs_weight_path):
                    continue
                
                for metric_name in all_metrics_to_compute:

                    # Form the round-join-metric tuple
                    round_join_metric_tuple = (train_dataset_name, model_architecture_name, 0, arrival_pattern, run_number, training_loop_name, al_method_name,
                                                al_budget, init_task_size, unl_buffer_size, round_id, current_epoch, dataset_split_path, model_weight_path, 
                                                opt_state_path, lr_state_path, completed_unix_time, metric_name, train_dataset_name, None, None)
                    metric_factory = MetricFactory(self.db_loc, self.base_exp_directory, self.base_dataset_directory, self.gpu_name, batch_size, round_join_metric_tuple, self.obj_det_config_path)
                    metric_to_compute = metric_factory.get_metric(metric_name)
                    metric_to_compute.evaluate()

                    add_metric_time_value_to_run(self.db_loc, train_dataset_name, model_architecture_name, 0, arrival_pattern, run_number, training_loop_name, 
                                                    al_method_name, al_budget, init_task_size, unl_buffer_size, round_id, metric_name, train_dataset_name, metric_to_compute.value, metric_to_compute.time)
                        
                exp_name_update = F"MET COMP {train_dataset_name:4s}_{model_architecture_name:4s}_{arrival_pattern:4s}_{run_number:4d}_{training_loop_name:4s}_{al_method_name:4s}_{al_budget:4d}_{init_task_size:4d}_{unl_buffer_size:4d}"
                self.notify_parent((self.gpu_name, exp_name_update, 1., round_id / num_rounds, 0))

            # After computing each metric for each round, we can delete the large objects associated with each round.
            for round_id, current_epoch, dataset_split_path, model_weight_path, opt_state_path, lr_state_path, completed_unix_time in round_info_list:
                head, tail = os.path.split(model_weight_path)
                abs_split_path, abs_weight_folder = get_absolute_paths_det(self.base_exp_directory, dataset_split_path, head)
                if os.path.exists(abs_split_path):      os.remove(abs_split_path)
                for file_name in os.listdir(abs_weight_folder):
                    abs_file_path = os.path.join(abs_weight_folder, file_name)
                    os.remove(abs_file_path)


    def do_al_round(self, train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size, num_rounds):
        
        # Get the most recent round
        full_train_dataset, test_transform, num_classes, \
        train_unlabeled_split, round_training_loop, model, prev_al_params, still_training = self.load_al_round(train_dataset_name, model_architecture_name, arrival_pattern, run_number, 
                                                                                                                training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size)

        # Get the task arrival pattern sequence information
        num_tasks = len(full_train_dataset.task_idx_partitions)
        if arrival_pattern == "sequential":
            task_arrival_pattern = sample_sequential_access_chain(num_tasks, num_rounds)
        elif arrival_pattern.startswith("rare_every"):
            every_mod = int(arrival_pattern.split("_")[2])
            task_arrival_pattern = sample_random_access_chain(num_tasks - 1, num_rounds)
            start_idx = 1 + every_mod
            while start_idx < len(task_arrival_pattern):
                task_arrival_pattern[start_idx] = num_tasks - 1
                start_idx = start_idx + every_mod
        else:
            raise ValueError("Unknown arrival pattern")

        # If the training loop has non-negative epoch, then the round is not finished. Do not select and proceed to training.
        if not still_training:
            
            # We are ready to select points for the new round, so advance the round number. If it is greater
            # than the number of rounds allowed, however, then return as the experiment has finished.
            self.round_number += 1
            if self.round_number >= num_rounds:
                return

            train_split_partitions      = train_unlabeled_split["train"]
            unlabeled_split_partitions  = train_unlabeled_split["unlabeled"]

            # Before creating the unlabeled dataset, we need to sample randomly from the unlabeled partition corresponding to the task
            # arrival pattern. SET REDUNDANCY FACTOR TO MATCH IN datasets/datasets.py
            redundancy_factor = 2   # SET TO MATCH
            arriving_task = task_arrival_pattern[self.round_number]
            unlabeled_buffer_size = min(len(unlabeled_split_partitions[arriving_task]), unl_buffer_size)
            randomly_chosen_task_unlabeled_idx = np.random.choice(unlabeled_split_partitions[arriving_task], size=unlabeled_buffer_size // redundancy_factor, replace=False).tolist()
            randomly_chosen_task_unlabeled_idx = randomly_chosen_task_unlabeled_idx * redundancy_factor
            round_unlabeled_task_idx_partitions = [[] for x in range(num_tasks)]
            round_unlabeled_task_idx_partitions[arriving_task] = randomly_chosen_task_unlabeled_idx

            train_dataset = TransformSubsetDetection(full_train_dataset, train_split_partitions)
            unlabeled_dataset = TransformSubsetDetection(full_train_dataset, round_unlabeled_task_idx_partitions)
            
            # Form the AL strategy.
            replay_buffer_capacity = init_task_size * num_tasks
            obj_det_config                  = Config.fromfile(self.obj_det_config_path)
            obj_det_config['device']        = self.gpu_name.split(":")[0]
            obj_det_config['gpu_ids']       = [int(self.gpu_name.split(":")[1])]
            obj_det_config["seed"]      = 0
            al_params = {"device": self.gpu_name, "batch_size": obj_det_config["data"]["samples_per_gpu"], "buffer_capacity": replay_buffer_capacity, "cfg": obj_det_config}
            if al_method_name.endswith("reservoir"):
                for al_param in prev_al_params:
                    if al_param.startswith("reservoir"):
                        al_params[al_param] = prev_al_params[al_param]
            elif "ulm_streamline" in al_method_name:
                al_params['acc_budget'] = prev_al_params['acc_budget'] if 'acc_budget' in prev_al_params else 0
            al_strategy_factory = ALFactory(train_dataset, unlabeled_dataset, model, num_classes, al_params)
            al_strategy = al_strategy_factory.get_strategy(al_method_name)

            # Do th AL selection. If this is a partition reservoir strategy, pass the task identity (oracle)
            selected_unlabeled_idx = al_strategy.select(al_budget)
            al_params               = al_strategy.args

            # selected_labeled_idx and selected_unlabeled_idx are lists of lists. Each one
            # has num_tasks lists, and the elements in those lists are indices with respect to the
            # named set (selected_<NAME>_idx). Hence, the zeroth list in selected_labeled_idx, for 
            # example, contains indices with respect to the labeled set that should belong to the
            # zeroth task. In case a task misidentification occurs, this means that indices from 
            # a different task may end up in the incorrect task partition.
            #
            # Here, we convert the indices so that they are with respect to the full dataset.
            # They remain within their newly assigned task (again, which are the same as their old 
            # task unless a task misidentification occurs). This will give us the new splits to use in 
            # training and the subsequent round.
            new_training_partitions = copy.deepcopy(train_split_partitions)
            for new_task_number, task_idx_partition_wrpt_unlabeled in enumerate(selected_unlabeled_idx):
                for idx_wrpt_unlabeled in task_idx_partition_wrpt_unlabeled:
                    old_task_number, within_index_partition = unlabeled_dataset.get_task_number_and_index_in_task(idx_wrpt_unlabeled)
                    full_labeled_idx = unlabeled_dataset.task_idx_partitions[old_task_number][within_index_partition]
                    new_training_partitions[new_task_number].append(full_labeled_idx)

            # Create the unlabeled idx partitions, which are simply that which isn't in the new training partitions.
            new_unlabeled_partitions = []
            for task_num in range(num_tasks):
                unlabeled_task_idx_partition = list(set(full_train_dataset.task_idx_partitions[task_num]) - set(new_training_partitions[task_num]))
                new_unlabeled_partitions.append(unlabeled_task_idx_partition)

            train_split = new_training_partitions
            unlabeled_split = new_unlabeled_partitions
            train_unlabeled_split["train"] =        train_split
            train_unlabeled_split["unlabeled"] =    unlabeled_split

            if "streamline" in al_method_name:
                for task_num, smi_base_fraction in enumerate(al_strategy.smi_base_fractions):
                    al_params[F"smi_base_fraction_{task_num}"] = smi_base_fraction

            # Get a training loop using the provided data split. Also, get a fresh model.
            train_split = train_unlabeled_split["train"]
            train_dataset = TransformSubsetDetection(full_train_dataset, train_split)  

            pretrain_weight_directory = os.path.join(self.base_exp_directory, "weights")
            model_factory = ModelFactory(num_classes=num_classes, pretrain_weight_directory=pretrain_weight_directory, obj_det_config_path=self.obj_det_config_path)
            new_model = model_factory.get_model(model_architecture_name)

            # Save the new split.
            split_path, working_dir, _      = generate_save_locations_for_al_round_det(train_dataset_name, model_architecture_name, 0, arrival_pattern, training_loop_name, al_method_name, run_number, al_budget, init_task_size, unl_buffer_size, self.round_number)
            abs_split_path, abs_working_dir = get_absolute_paths_det(self.base_exp_directory, split_path, working_dir)
            atomic_json_save(abs_split_path, train_unlabeled_split)

            # Get a training loop using the (new) training split
            obj_det_config                  = Config.fromfile(self.obj_det_config_path)
            obj_det_config['device']        = self.gpu_name.split(":")[0]
            obj_det_config['gpu_ids']       = [int(self.gpu_name.split(":")[1])]
            obj_det_config["work_dir"]      = abs_working_dir
            obj_det_config["resume_from"]   = None
            obj_det_config["seed"]      = 0
            training_loop_factory = TrainingLoopFactory(train_dataset, test_transform, new_model, None, obj_det_config)
            round_training_loop = training_loop_factory.get_training_loop(training_loop_name)

        else:

            al_params = prev_al_params

        # Train using mmdetection's training utility. It will also take care of checkpointing.
        round_training_loop.train()

        # Save the results (no metrics or AL parameters currently)
        self.save_al_round(train_unlabeled_split, train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name,
                           al_method_name, al_budget, init_task_size, unl_buffer_size, self.round_number, al_params, test_transform, [])


    def save_al_round(self, train_unlabeled_dataset_split, train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, 
                      al_method_name, al_budget, init_task_size, unl_buffer_size, round_number, al_params, test_augmentation, metrics):
 
        # Get work directory for mmdetection. Save the split info to the abs path.
        split_path, working_dir, completion_unix_time   = generate_save_locations_for_al_round_det(train_dataset_name, model_architecture_name, 0, arrival_pattern, training_loop_name, al_method_name, run_number, al_budget, init_task_size, unl_buffer_size, self.round_number)
        abs_split_path, abs_working_dir                 = get_absolute_paths_det(self.base_exp_directory, split_path, working_dir)
        atomic_json_save(abs_split_path, train_unlabeled_dataset_split)

        # Format the al params, keeping only those with integer/real values
        save_al_params = list()
        for al_param in al_params:
            if type(al_params[al_param]) == list:
                for i, value in enumerate(al_params[al_param]):
                    save_al_params.append((F"{al_param}_{i}", value))
            else:
                if type(al_params[al_param]) != int and type(al_params[al_param]) != float:
                    continue
                save_al_params.append((al_param, al_params[al_param]))

        # Add record of AL round to the database. We set the following values as defaults for this experiment:
        #   1. epoch: -1                (to show that the training has finished)
        #   2. limited_mem: 1           (per exp definition)
        #   3. augmentation: []         (hard-coded in config)
        #   4. stop_criteria:       []  (hard-coded in config)
        #   5. opt/lr_state:    ""      (not used / handled by mmdet)
        rel_weight_path = os.path.join(working_dir, "latest.pth")
        add_al_round(self.db_loc, train_dataset_name, model_architecture_name, 0, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, round_number,
                     -1, split_path, rel_weight_path, "", "", completion_unix_time, save_al_params, [], [], [], metrics)

        # As a last step, remove all the checkpoints in the working folder that aren't the latest.pth file or its symbolic link.
        all_saved_weights = os.listdir(abs_working_dir)
        latest_epoch = -1
        for filename in all_saved_weights:
            if filename.startswith("epoch_"):
                epoch_num       = int(filename.split("_")[1].split(".pth")[0])
                latest_epoch    = max(epoch_num, latest_epoch)

        for filename in all_saved_weights:
            if filename != "latest.pth" and filename != F"epoch_{latest_epoch}.pth":
                abs_filename = os.path.join(abs_working_dir, filename)
                os.remove(abs_filename)

    
    def load_al_round(self, train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size):
    
        # Get the most recent round info, some fields of which will not be useful in instantiating / resuming the experiment
        round_info_tuple = get_most_recent_round_info(self.db_loc, train_dataset_name, model_architecture_name, 0, arrival_pattern,
                                                      run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size)

        if round_info_tuple is not None:
            
            # There exists completed rounds. Go ahead and unpack the tuple to get the round number. Advance it by 1 to get the round we wish to load.    
            _, _, _, _, _, _, _, _, _, _, self.round_number, _, _, _, _, _, _, al_params, _, _, _ = round_info_tuple
            self.round_number += 1
        
        else:
            
            # There are no completed rounds. Start with the initial round.
            self.round_number = 0
            al_params = dict()

        # Because the detection experiments do not save round info until the end of the round (as checkpointing is managed
        # by mmdetection), there may be a round currently active. We can check if a round is active by checking for model info.
        split_path, working_dir, _ = generate_save_locations_for_al_round_det(train_dataset_name, model_architecture_name, 0, arrival_pattern, training_loop_name, al_method_name, run_number, al_budget, init_task_size, unl_buffer_size, self.round_number)
        abs_split_path, abs_working_dir = get_absolute_paths_det(self.base_exp_directory, split_path, working_dir)
        abs_latest_chkpt_path = os.path.join(abs_working_dir, "latest.pth")
        still_training = True

        if not os.path.exists(abs_latest_chkpt_path):

            # If this is also the initial round (with no model info), it means this is the very first time the run is
            # being executed. Go ahead and create the initial round.
            if self.round_number == 0:
                return self.create_initial_round(train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size)
            else:
                # Otherwise, this is the first time the *round* is being run. For selection purposes, we need to reinstate the previous round.
                self.round_number -= 1
                still_training = False

                split_path, working_dir, _      = generate_save_locations_for_al_round_det(train_dataset_name, model_architecture_name, 0, arrival_pattern, training_loop_name, al_method_name, run_number, al_budget, init_task_size, unl_buffer_size, self.round_number)
                abs_split_path, abs_working_dir = get_absolute_paths_det(self.base_exp_directory, split_path, working_dir)
                abs_latest_chkpt_path           = os.path.join(abs_working_dir, "latest.pth")

        # BDD100K requires different treatment. We will use MMDetection to load the base dataset based on the configs given in the utils folder.
        obj_det_config                  = Config.fromfile(self.obj_det_config_path)
        obj_det_config['device']        = self.gpu_name.split(":")[0]
        obj_det_config['gpu_ids']       = [int(self.gpu_name.split(":")[1])]
        obj_det_config["work_dir"]      = abs_working_dir
        obj_det_config["resume_from"]   = abs_latest_chkpt_path
        obj_det_config["seed"]      = 0

        # Retrieve the split data
        with open(abs_split_path, "r") as f:
            train_unlabeled_dataset_split = json.load(f)

        # Get the dataset
        dataset_factory = DatasetFactory(self.base_dataset_directory)
        full_train_dataset, test_transform, num_classes = dataset_factory.get_dataset(train_dataset_name)

        # Get the model. However, if there is a checkpoint we need to be loading from, initialize the model with those weights.
        pretrain_weight_directory = os.path.join(self.base_exp_directory, "weights")
        model_factory = ModelFactory(num_classes=num_classes, pretrain_weight_directory=pretrain_weight_directory, obj_det_config_path=self.obj_det_config_path)
        model = model_factory.get_model(model_architecture_name)

        if os.path.exists(abs_latest_chkpt_path):
            checkpoint      = load_checkpoint(model, abs_latest_chkpt_path, map_location="cpu")
            model.cfg       = obj_det_config
        
        model.to(self.gpu_name)
        model.eval()
        
        # Get a training loop using the provided data split
        train_split = train_unlabeled_dataset_split["train"]
        train_dataset = TransformSubsetDetection(full_train_dataset, train_split)  

        # BDD100K requires different treatment. We will use MMDetection to load the base dataset based on the configs given in the utils folder.
        training_loop_factory = TrainingLoopFactory(train_dataset, test_transform, model, None, obj_det_config)
        round_training_loop = training_loop_factory.get_training_loop(training_loop_name)

        # Lastly, take care of the the al params array.
        new_al_params = dict()
        for name, value in al_params:
            new_al_params[name] = value
        
        # Restructure as a list if there is reservoir_counters_0
        if "reservoir_counters_0" in new_al_params:
            num_tasks = len(full_train_dataset.task_idx_partitions)
            reservoir_counters_list = [0 for x in range(num_tasks)]
            for key in list(new_al_params.keys()):
                if key.startswith("reservoir_counters_"):
                    counter_num = int(key.split("_")[2])
                    reservoir_counters_list[counter_num] = new_al_params[key]
                    del new_al_params[key]
            new_al_params["reservoir_counters"] = reservoir_counters_list
        al_params = new_al_params

        return full_train_dataset, test_transform, num_classes, train_unlabeled_dataset_split, round_training_loop, model, al_params, still_training


    def create_initial_round(self, train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size):
        
        self.round_number = 0
        
        # Get the dataset
        dataset_factory = DatasetFactory(self.base_dataset_directory)
        full_train_dataset, test_transform, num_classes = dataset_factory.get_dataset(train_dataset_name)
        train_split, unlabeled_split = dataset_factory.get_initial_split(train_dataset_name, init_task_size)
        train_unlabeled_dataset_split = {"train": train_split, "unlabeled": unlabeled_split}
        
        # Get the model
        pretrain_weight_directory = os.path.join(self.base_exp_directory, "weights")
        model_factory = ModelFactory(num_classes=num_classes, pretrain_weight_directory=pretrain_weight_directory, obj_det_config_path=self.obj_det_config_path)
        model = model_factory.get_model(model_architecture_name)
        
        # Get a training loop using the provided data split
        train_split = train_unlabeled_dataset_split["train"]
        train_dataset = TransformSubsetDetection(full_train_dataset, train_split)  

        # Get work directory for mmdetection. Save the split info to the abs path.
        split_path, working_dir, _ = generate_save_locations_for_al_round_det(train_dataset_name, model_architecture_name, 0, arrival_pattern, training_loop_name, al_method_name, run_number, al_budget, init_task_size, unl_buffer_size, self.round_number)
        abs_split_path, abs_working_dir = get_absolute_paths_det(self.base_exp_directory, split_path, working_dir)
        atomic_json_save(abs_split_path, train_unlabeled_dataset_split)

        # BDD100K requires different treatment. We will use MMDetection to load the base dataset based on the configs given in the utils folder.
        obj_det_config                  = Config.fromfile(self.obj_det_config_path)
        obj_det_config['device']        = self.gpu_name.split(":")[0]
        obj_det_config['gpu_ids']       = [int(self.gpu_name.split(":")[1])]
        obj_det_config["work_dir"]      = abs_working_dir
        obj_det_config["resume_from"]   = None
        obj_det_config["seed"]      = 0
        training_loop_factory = TrainingLoopFactory(train_dataset, test_transform, model, None, obj_det_config)
        round_training_loop = training_loop_factory.get_training_loop(training_loop_name)

        return full_train_dataset, test_transform, num_classes, train_unlabeled_dataset_split, round_training_loop, model, dict(), True