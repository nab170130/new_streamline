import copy
from .experiment import Experiment
from ..al_strategies import ALFactory
from ..datasets import DatasetFactory, TransformSubset
from ..metrics import MetricFactory
from ..models import ModelFactory
from ..persistence import add_al_round, format_augmentation, generate_save_locations_for_al_round, get_absolute_paths, get_most_recent_round_info, atomic_torch_save, atomic_json_save, get_all_rounds_from_run, add_metric_time_value_to_run
from ..training_loops import TrainingLoopFactory
from ..utils import sample_sequential_access_chain, sample_rare_access_chain, sample_random_access_chain

import json
import numpy as np
import os
import torch

class UnlimitedMemoryExperiment(Experiment):
    

    def __init__(self, comm_lock, comm_sem, parent_pipe, gpu_name, base_dataset_directory, base_exp_directory, db_loc):

        super(UnlimitedMemoryExperiment, self).__init__(comm_lock, comm_sem, parent_pipe, gpu_name, base_dataset_directory, base_exp_directory, db_loc)
        self.round_number = 0


    def experiment(self, train_dataset_name, model_architecture_name, limited_memory, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size, num_rounds, delete_large_objects_after_run=False):

        # Get all current rounds. In the event that the epoch is -1, skip the round.
        try:
            round_info_list = get_all_rounds_from_run(self.db_loc, train_dataset_name, model_architecture_name, 0, arrival_pattern, run_number,
                                                        training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size)
        except ValueError:
            round_info_list = []

        self.num_rounds = num_rounds
        while self.round_number < num_rounds:
            if self.round_number < len(round_info_list):
                curr_epoch_index = 1
                curr_epoch = round_info_list[self.round_number][curr_epoch_index]
                if curr_epoch < 0:
                    self.round_number += 1
                    continue

            self.do_al_round(train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size, num_rounds)

        if delete_large_objects_after_run:

            # If the large objects are going to be deleted, then calculate each metric before deletion.
            round_info_list = get_all_rounds_from_run(self.db_loc, train_dataset_name, model_architecture_name, 0, arrival_pattern, run_number,
                                                        training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size)

            # Get the number of tasks for the dataset and all the metrics to compute.
            num_tasks = len(DatasetFactory(self.base_dataset_directory).get_dataset(train_dataset_name)[0].task_idx_partitions)
            all_metrics_to_compute = ["accuracy", "labeled_instances", "task_identification_accuracy"]
            for task in range(num_tasks):
                all_metrics_to_compute.append(F"task_presence_{task}")
                all_metrics_to_compute.append(F"per_task_accuracy_{task}")
            
            for round_id, current_epoch, dataset_split_path, model_weight_path, opt_state_path, lr_state_path, completed_unix_time in round_info_list:
                
                # Don't do anything if there are no large objects. This means that the metrics should have already been computed.
                abs_split_path, abs_weight_path, abs_opt_state_path, abs_lr_state_path = get_absolute_paths(self.base_exp_directory, dataset_split_path, model_weight_path, opt_state_path, lr_state_path)
                if not os.path.exists(abs_split_path):
                    continue
                if not os.path.exists(abs_weight_path):
                    continue
                if not os.path.exists(abs_opt_state_path):
                    continue
                if not os.path.exists(abs_lr_state_path):
                    continue
                
                for metric_name in all_metrics_to_compute:

                    # Form the round-join-metric tuple
                    round_join_metric_tuple = (train_dataset_name, model_architecture_name, 0, arrival_pattern, run_number, training_loop_name, al_method_name,
                                                al_budget, init_task_size, unl_buffer_size, round_id, current_epoch, dataset_split_path, model_weight_path, 
                                                opt_state_path, lr_state_path, completed_unix_time, metric_name, train_dataset_name, None, None)
                    metric_factory = MetricFactory(self.db_loc, self.base_exp_directory, self.base_dataset_directory, self.gpu_name, batch_size, round_join_metric_tuple)
                    metric_to_compute = metric_factory.get_metric(metric_name)
                    metric_to_compute.evaluate()

                    add_metric_time_value_to_run(self.db_loc, train_dataset_name, model_architecture_name, 0, arrival_pattern, run_number, training_loop_name, 
                                                    al_method_name, al_budget, init_task_size, unl_buffer_size, round_id, metric_name, train_dataset_name, metric_to_compute.value, metric_to_compute.time)
                        
                exp_name_update = F"MET COMP {train_dataset_name:4s}_{model_architecture_name:4s}_{arrival_pattern:4s}_{run_number:4d}_{training_loop_name:4s}_{al_method_name:4s}_{al_budget:4d}_{init_task_size:4d}_{unl_buffer_size:4d}"
                self.notify_parent((self.gpu_name, exp_name_update, 1., round_id / num_rounds, 0))

            # After computing each metric for each round, we can delete the large objects associated with each round.
            for round_id, current_epoch, dataset_split_path, model_weight_path, opt_state_path, lr_state_path, completed_unix_time in round_info_list:
                abs_split_path, abs_weight_path, abs_opt_state_path, abs_lr_state_path = get_absolute_paths(self.base_exp_directory, dataset_split_path, model_weight_path, opt_state_path, lr_state_path)
                if os.path.exists(abs_split_path):      os.remove(abs_split_path)
                if os.path.exists(abs_weight_path):     os.remove(abs_weight_path)
                if os.path.exists(abs_opt_state_path):  os.remove(abs_opt_state_path)
                if os.path.exists(abs_lr_state_path):   os.remove(abs_lr_state_path)


    def do_al_round(self, train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size, num_rounds):
        
        # Get the most recent round
        full_train_dataset, test_transform, num_classes, \
        train_unlabeled_split, round_training_loop, model, prev_al_params = self.load_al_round(train_dataset_name, model_architecture_name, arrival_pattern, run_number, 
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
        print("TASK PATTERN", task_arrival_pattern)

        # If the training loop has non-negative epoch, then the round is not finished. Do not select and proceed to training.
        if round_training_loop.epoch < 0:
            
            # We are ready to select points for the new round, so advance the round number. If it is greater
            # than the number of rounds allowed, however, then return as the experiment has finished.
            self.round_number += 1
            if self.round_number >= num_rounds:
                return

            train_split_partitions = train_unlabeled_split["train"]
            unlabeled_split_partitions = train_unlabeled_split["unlabeled"]

            # Before creating the unlabeled dataset, we need to sample randomly from the unlabeled partition corresponding to the task
            # arrival pattern.
            arriving_task = task_arrival_pattern[self.round_number]
            unlabeled_buffer_size = min(len(unlabeled_split_partitions[arriving_task]), unl_buffer_size)
            randomly_chosen_task_unlabeled_idx = np.random.choice(unlabeled_split_partitions[arriving_task], size=unlabeled_buffer_size, replace=False).tolist()
            round_unlabeled_task_idx_partitions = [[] for x in range(num_tasks)]
            round_unlabeled_task_idx_partitions[arriving_task] = randomly_chosen_task_unlabeled_idx

            train_dataset = TransformSubset(full_train_dataset, train_split_partitions, is_labeled=True, transform=test_transform)
            unlabeled_dataset = TransformSubset(full_train_dataset, round_unlabeled_task_idx_partitions, is_labeled=False, transform=test_transform)
            
            # Form the AL strategy.
            replay_buffer_capacity = init_task_size * num_tasks
            al_params = {"device": self.gpu_name, "batch_size": batch_size, "lr": round_training_loop.optimizer.param_groups[0]['lr'], "buffer_capacity": replay_buffer_capacity}
            
            # Tune the min-budget factor depending on the dataset.  FOR ABLATION EXPERIMENTS, CHANGE THIS TO 0.5.
            if train_dataset_name == "PovertyMap":
                al_params["min_budget"] = 0.825
                
            if al_method_name.endswith("reservoir"):
                for al_param in prev_al_params:
                    if al_param.startswith("reservoir"):
                        al_params[al_param] = prev_al_params[al_param]
            elif "ulm_streamline" in al_method_name:
                al_params['acc_budget'] = prev_al_params['acc_budget'] if 'acc_budget' in prev_al_params else 0
            al_strategy_factory = ALFactory(train_dataset, unlabeled_dataset, model, num_classes, al_params)
            al_strategy = al_strategy_factory.get_strategy(al_method_name)

            # Do th AL selection.
            selected_unlabeled_idx  = al_strategy.select(al_budget)
            al_params               = al_strategy.args

            # selected_unlabeled_idx is a list of lists. It has num_tasks lists, and the elements 
            # in those lists are indices with respect to the unlabeled set. Hence, the zeroth list in 
            # selected_labeled_idx, for example, contains indices with respect to the labeled set 
            # that should belong to the zeroth task. In case a task misidentification occurs, this means 
            # that indices from a different task may end up in the incorrect task partition.
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

            # Get a training loop using the (new) training split
            train_callback = UnlimitedMemoryExperimentCallback(train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, \
                                                                self.round_number, al_params, train_unlabeled_split, test_transform, self.num_rounds, self.gpu_name, self.notify_parent, self.save_al_round)
            train_dataset = TransformSubset(full_train_dataset, train_split)
            train_args = {"device": self.gpu_name, "batch_size": batch_size}
            training_loop_factory = TrainingLoopFactory(train_dataset, test_transform, model, train_callback, train_args)
            round_training_loop = training_loop_factory.get_training_loop(training_loop_name)

        else:

            al_params = prev_al_params

        # Train using the factory-constructed loop
        new_model = round_training_loop.train()
        
        # Save the results (no metrics or AL parameters currently)
        self.save_al_round(train_unlabeled_split, train_dataset_name, model_architecture_name, arrival_pattern, run_number, round_training_loop,
                           al_method_name, al_budget, init_task_size, unl_buffer_size, self.round_number, al_params, test_transform, [])


    def save_al_round(self, train_unlabeled_val_dataset_split, train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop, 
                      al_method_name, al_budget, init_task_size, unl_buffer_size, round_number, al_params, test_augmentation, metrics):
    
        # Extract training information from the training loop
        model, training_loop_name, opt_state_dict, lr_state_dict, stop_criteria, train_augmentation, epoch = training_loop.extract_training_details()
        lr_state_dict = "" if lr_state_dict is None else lr_state_dict
        train_augmentation_as_items = format_augmentation(train_augmentation)
        
        # Get saving paths for the state info
        split_path, weight_path, opt_state_path, lr_state_path, completion_unix_time = generate_save_locations_for_al_round(train_dataset_name, model_architecture_name, 0, arrival_pattern, training_loop_name, al_method_name, run_number, al_budget, init_task_size, unl_buffer_size, round_number)
        
        # Format the test
        test_augmentation_as_items = format_augmentation(test_augmentation)

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

        # Save state information stored in dictionaries
        abs_split_path, abs_weight_path, abs_opt_state_path, abs_lr_state_path = get_absolute_paths(self.base_exp_directory, split_path, weight_path, opt_state_path, lr_state_path)
        atomic_json_save(abs_split_path, train_unlabeled_val_dataset_split)
        atomic_torch_save(abs_weight_path, model.state_dict())
        atomic_torch_save(abs_opt_state_path, opt_state_dict)      
        if lr_state_dict != "":
            atomic_torch_save(abs_lr_state_path, lr_state_dict)
        
        # Add record of AL round to the database
        add_al_round(self.db_loc, train_dataset_name, model_architecture_name, 0, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, round_number,
                     epoch, split_path, weight_path, opt_state_path, lr_state_path, completion_unix_time, save_al_params, train_augmentation_as_items,
                     test_augmentation_as_items, stop_criteria, metrics)
    
    
    def load_al_round(self, train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size):
    
        # Get the most recent round info, some fields of which will not be useful in instantiating / resuming the experiment
        round_info_tuple = get_most_recent_round_info(self.db_loc, train_dataset_name, model_architecture_name, 0, arrival_pattern,
                                                      run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size)
        
        if round_info_tuple is None:
            return self.create_initial_round(train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size)
        else:

            # Unpack the tuple         
            _, _, _, _, _, _, _, _, _, _, self.round_number, epoch, split_path, weight_path, opt_state_path, lr_state_path, _, al_params, _, _, _ = round_info_tuple
            
            # Get absolute locations from which to load
            abs_split_path, abs_weight_path, abs_opt_state_path, abs_lr_state_path = get_absolute_paths(self.base_exp_directory, split_path, weight_path, opt_state_path, lr_state_path)
            
            # Retrieve the split data
            with open(abs_split_path, "r") as f:
                train_unlabeled_dataset_split = json.load(f)
            
            # Retrieve model weights, optimizer state, learning rate state
            with open(abs_weight_path, "rb") as f:
                model_state_dict = torch.load(f, map_location=torch.device("cpu"))
            
            with open(abs_opt_state_path, "rb") as f:
                opt_state_dict = torch.load(f, map_location=torch.device("cpu"))

            if lr_state_path != "":
                with open(abs_lr_state_path, "rb") as f:
                    lr_state_dict = torch.load(f, map_location=torch.device("cpu"))
            else:
                lr_state_dict = None

            # Get the dataset
            dataset_factory = DatasetFactory(self.base_dataset_directory)
            full_train_dataset, test_transform, num_classes = dataset_factory.get_dataset(train_dataset_name)

            # Get the model
            pretrain_weight_directory = os.path.join(self.base_exp_directory, "weights")
            model_factory = ModelFactory(num_classes=num_classes, pretrain_weight_directory=pretrain_weight_directory)
            model = model_factory.get_model(model_architecture_name)
            model.load_state_dict(model_state_dict)

            # Reformat the al params
            new_al_params = dict()
            for name, value in al_params:
                new_al_params[name] = value
            
            # Restructure as a list if there is reservoir_counters_0
            al_params = new_al_params

            # Get a training loop using the provided data split
            train_split = train_unlabeled_dataset_split["train"]
            train_dataset = TransformSubset(full_train_dataset, train_split)
            train_callback = UnlimitedMemoryExperimentCallback(train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, \
                                                                self.round_number, al_params, train_unlabeled_dataset_split, test_transform, self.num_rounds, self.gpu_name, self.notify_parent, self.save_al_round)

            train_args = {"device": self.gpu_name, "batch_size": batch_size}
            training_loop_factory = TrainingLoopFactory(train_dataset, test_transform, model, train_callback, train_args)
            round_training_loop = training_loop_factory.get_training_loop(training_loop_name)
            round_training_loop.from_checkpoint(epoch, model_state_dict, opt_state_dict, lr_state_dict)

            return full_train_dataset, test_transform, num_classes, train_unlabeled_dataset_split, round_training_loop, model, al_params


    def create_initial_round(self, train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, batch_size):
        
        self.round_number = 0
        
        # Get the dataset
        dataset_factory = DatasetFactory(self.base_dataset_directory)
        full_train_dataset, test_transform, num_classes = dataset_factory.get_dataset(train_dataset_name)
        train_split, unlabeled_split = dataset_factory.get_initial_split(train_dataset_name, init_task_size)
        train_unlabeled_dataset_split = {"train": train_split, "unlabeled": unlabeled_split}
        
        # Get the model
        pretrain_weight_directory = os.path.join(self.base_exp_directory, "weights")
        model_factory = ModelFactory(num_classes=num_classes, pretrain_weight_directory=pretrain_weight_directory)
        model = model_factory.get_model(model_architecture_name)
        
        # Get a training loop using the provided data split
        train_split = train_unlabeled_dataset_split["train"]
        train_dataset = TransformSubset(full_train_dataset, train_split)

        train_callback = UnlimitedMemoryExperimentCallback(train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, \
                                                            self.round_number, dict(), train_unlabeled_dataset_split, test_transform, self.num_rounds, self.gpu_name, self.notify_parent, self.save_al_round)
        train_args = {"device": self.gpu_name, "batch_size": batch_size}
        training_loop_factory = TrainingLoopFactory(train_dataset, test_transform, model, train_callback, train_args)
        round_training_loop = training_loop_factory.get_training_loop(training_loop_name)

        return full_train_dataset, test_transform, num_classes, train_unlabeled_dataset_split, round_training_loop, model, dict()


class UnlimitedMemoryExperimentCallback:
        

        def __init__(self, train_dataset_name, model_architecture_name, arrival_pattern, run_number, training_loop_name, al_method_name, al_budget, init_task_size, unl_buffer_size, \
                     round_number, al_params, train_unlabeled_dataset_split, test_transform, num_rounds, gpu_name, message_function, save_function):
            
            self.train_dataset_name = train_dataset_name
            self.model_architecture_name = model_architecture_name
            self.arrival_pattern = arrival_pattern
            self.training_loop_name = training_loop_name
            self.al_method_name = al_method_name
            self.run_number = run_number
            self.al_budget = al_budget
            self.init_task_size = init_task_size
            self.unl_buffer_size = unl_buffer_size
            self.round_number = round_number
            self.al_params = al_params
            self.train_unlabeled_dataset_split = train_unlabeled_dataset_split
            self.test_transform = test_transform
            self.run_progress = round_number / num_rounds
            self.gpu_name = gpu_name
            self.message_function = message_function
            self.save_function = save_function
            self.name = F"{train_dataset_name:4s}_{model_architecture_name:4s}_{arrival_pattern:4s}_{run_number:4d}_{training_loop_name:4s}_{al_method_name:4s}_{al_budget:4d}_{init_task_size:4d}_{unl_buffer_size:4d}"
            
            
        def __call__(self, training_loop, round_progress, current_accuracy):
            
            manager_message = (self.gpu_name, self.name, self.run_progress, round_progress, current_accuracy)
            self.message_function(manager_message)

            if training_loop.epoch % 5 == 0:
                self.save_function(self.train_unlabeled_dataset_split, self.train_dataset_name, self.model_architecture_name, self.arrival_pattern, self.run_number, 
                                    training_loop, self.al_method_name, self.al_budget, self.init_task_size, self.unl_buffer_size, self.round_number, self.al_params, self.test_transform, [])