import json
import os
import time

import torch

def format_augmentation(augmentation):
        
    augmentation_as_items = []
    for individual_transform in augmentation.transforms:
        descriptor = individual_transform.__str__()
        augmentation_as_items.append(descriptor)

    return augmentation_as_items


def generate_save_locations_for_al_round(train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, training_loop, al_method_name, run_number, al_budget, init_task_size, unl_buffer_size, round_number):
        
    # Mark a save location with the current unix time, which is practically guaranteed to not produce conflicts
    current_unix_time = time.time_ns()
    
    round_identifier = F"{train_dataset_name}_{model_architecture_name}_{limited_mem}_{arrival_pattern}_{run_number}_{training_loop}_{al_method_name}_{al_budget}_{init_task_size}_{unl_buffer_size}_{round_number}"

    split_path = F"split_{round_identifier}.json"
    weight_path = F"weight_{round_identifier}.pth"
    opt_state_path = F"opt_state_{round_identifier}.pth"
    lr_state_path = F"lr_state_{round_identifier}.pth"
    
    return split_path, weight_path, opt_state_path, lr_state_path, current_unix_time


def get_absolute_paths(base_exp_directory, split_path, weight_path, opt_state_path, lr_state_path):
    
    # Attempt to create directories for splits, models weights, optimizer states, and lr scheduler states
    split_base_dir = os.path.join(base_exp_directory, "splits")
    weight_base_dir = os.path.join(base_exp_directory, "weights")
    opt_state_base_dir = os.path.join(base_exp_directory, "opt_states")
    lr_state_base_dir = os.path.join(base_exp_directory, "lr_states")
    
    os.makedirs(split_base_dir, exist_ok=True)
    os.makedirs(weight_base_dir, exist_ok=True)
    os.makedirs(opt_state_base_dir, exist_ok=True)
    os.makedirs(lr_state_base_dir, exist_ok=True)
    
    abs_split_path = os.path.join(split_base_dir, split_path)
    abs_weight_path = os.path.join(weight_base_dir, weight_path)
    abs_opt_state_path = os.path.join(opt_state_base_dir, opt_state_path)
    abs_lr_state_path = os.path.join(lr_state_base_dir, lr_state_path)
    
    return abs_split_path, abs_weight_path, abs_opt_state_path, abs_lr_state_path


def generate_save_locations_for_al_round_det(train_dataset_name, model_architecture_name, limited_mem, arrival_pattern, training_loop, al_method_name, run_number, al_budget, init_task_size, unl_buffer_size, round_number):

    # Mark a save location with the current unix time, which is practically guaranteed to not produce conflicts
    current_unix_time = time.time_ns()
    
    round_identifier = F"{train_dataset_name}_{model_architecture_name}_{limited_mem}_{arrival_pattern}_{run_number}_{training_loop}_{al_method_name}_{al_budget}_{init_task_size}_{unl_buffer_size}_{round_number}"

    split_path      = os.path.join(F"split_{round_identifier}.json")
    weight_folder   = os.path.join(round_identifier) 
    
    return split_path, weight_folder, current_unix_time


def get_absolute_paths_det(base_exp_directory, split_path, weight_folder):
    
    # Attempt to create directories for splits, models weights, optimizer states, and lr scheduler states
    split_base_dir = os.path.join(base_exp_directory, "splits")
    weight_base_dir = os.path.join(base_exp_directory, "weights")
    
    os.makedirs(split_base_dir, exist_ok=True)
    os.makedirs(weight_base_dir, exist_ok=True)
    
    abs_split_path = os.path.join(split_base_dir, split_path)
    abs_weight_path = os.path.join(weight_base_dir, weight_folder)

    os.makedirs(abs_weight_path, exist_ok=True)
    
    return abs_split_path, abs_weight_path


def _get_temp_path(abs_loc):

    save_directory, file_name = os.path.split(abs_loc)
    temp_loc = os.path.join(save_directory, F"temp_{time.time_ns()}")
    return temp_loc


def atomic_json_save(abs_loc, json_obj):
    
    temp_loc = _get_temp_path(abs_loc)
    with open(temp_loc, "w") as f:
        json.dump(json_obj, f)
    os.rename(temp_loc, abs_loc)


def atomic_torch_save(abs_loc, torch_obj):
    
    temp_loc = _get_temp_path(abs_loc)
    torch.save(torch_obj, temp_loc)
    os.rename(temp_loc, abs_loc)