from streamline.persistence import create_db

import argparse
import json
import multiprocessing as mp
import os
import threading as th
import sys


def get_progress_bar_string(gpu_name, exp_name, run_progress, round_progress, current_accuracy, bar_size = 20):
    
    BAR_CHAR = "="

    run_progress_fill = int(bar_size * run_progress)
    run_progress_empty = bar_size - run_progress_fill
    round_progress_fill = int(bar_size * round_progress)
    round_progress_empty = bar_size - round_progress_fill 
    
    run_progress_bar = "[" + BAR_CHAR * run_progress_fill + " " * run_progress_empty + "]"
    round_progress_bar = "[" + BAR_CHAR * round_progress_fill + " " * round_progress_empty + "]"
    full_progress_bar_string = F"{exp_name:30s} ({gpu_name:6s}): (RUN COMPLETION: {run_progress_bar} {run_progress:1.2%}) (ROUND COMPLETION: {round_progress_bar} {round_progress:1.2%} (Acc: {current_accuracy:1.2%}))"
    return full_progress_bar_string    


def get_full_status_string(progress_strings, vert_shift_amt):
    
    max_row_len = -float('inf')
    for gpu_name in progress_strings:
        max_row_len = max(max_row_len, len(progress_strings[gpu_name]))
    max_row_len = int(max_row_len)
    
    if False: #vert_shift_amt > 0:
        full_status_string = F"\u001b[{vert_shift_amt}A"
    else:
        full_status_string = ""
        
    full_status_string += "=" * max_row_len + "\n"
    for gpu_name in progress_strings:
        full_status_string += progress_strings[gpu_name] + "\n"
    full_status_string += "=" * max_row_len
    
    future_shift_amt = 2 + len(progress_strings.keys())
    
    return full_status_string, future_shift_amt


def print_thread_entry(recv_pipe, wait_sem, gpu_list):
    
    # Enable ANSI codes
    os.system("color")
    
    progress_strings = {}
    shift_amt = 0
    
    # Keep receiving messages from workers, which update status
    # Keep updating panel until the manager terminates (this 
    # thread is started as a daemon thread).
    while True:
        gpu_name, exp_name, run_progress, round_progress, current_accuracy = recv_pipe.recv()
        wait_sem.release()

        progress_strings[gpu_name] = get_progress_bar_string(gpu_name, exp_name, run_progress, round_progress, current_accuracy)
        full_status_string, shift_amt = get_full_status_string(progress_strings, shift_amt)
        print(full_status_string)


def _worker_entry(experiment, experiment_arguments, assigned_gpu, gpu_availability_queue, distil_loc=None):
    
    # Add DISTIL location to path variable if needed
    if distil_loc is not None:
        sys.path.append(distil_loc)

    # Run the experiment's main code
    experiment.experiment(*experiment_arguments)

    # Add the GPU back to the queue
    gpu_availability_queue.put(assigned_gpu)


if __name__ == "__main__":
    
    # Set start method to spawn(). CUDA cannot be initialized again in forked
    # subprocesses.
    mp.set_start_method("spawn")

    # Retrieve the experiment configuration from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_db", action=argparse.BooleanOptionalAction, help="Whether a DB should be created")
    parser.add_argument("exp_config", type=str, help="Location of config file (json)")
    args = parser.parse_args()
    json_config_loc = args.exp_config
    should_create_db = args.create_db
    with open(json_config_loc, "r") as f:
        experiment_config = json.load(f)
    
    # Get DISTIL's location if needed/provided
    if "distil_directory" in experiment_config:
        distil_loc = experiment_config["distil_directory"]
        sys.path.append(distil_loc)
    else:
        distil_loc = None

    # Get the list of GPUs that should be used.
    # Note that we make the assumption that one
    # process requires one full GPU. GPUs are 
    # assigned based on a simple queueing mechanism.
    gpu_list = experiment_config["gpus"]
    n_proc = len(gpu_list)
    gpu_availability_queue = mp.Queue()
    for gpu in gpu_list:
        gpu_availability_queue.put(gpu)
    
    # Get saving locations
    base_dataset_directory = experiment_config["dataset_directory"]
    base_exp_directory = experiment_config["base_exp_directory"]
    db_loc = experiment_config["db_loc"]

    if should_create_db:
        if not os.path.exists(db_loc):
            create_db(db_loc)
    
    # Initialize worker-manager synchronization primitives
    worker_lock = mp.Lock()
    worker_sem = mp.Semaphore(0)
    
    # Create a worker-manager pipe and start the manager receiver thread
    manager_side_pipe, worker_side_pipe = mp.Pipe()
    parent_print_thread = th.Thread(target=print_thread_entry, args=(manager_side_pipe, worker_sem, gpu_list), daemon=True)
    parent_print_thread.start()    
    
    # Get the list of experiment configurations, which are arranged as tuples
    all_spawned_processes = []
    for experiment_arguments in experiment_config["experiments"]:

        # Stop if there are no GPUs available. If so, wait until one of the child processes finishes.
        if gpu_availability_queue.empty():
            os.waitpid(-1, 0)
        
        # Pick a GPU out of the queue
        gpu_to_assign = gpu_availability_queue.get()
        
        # Decipher which experiment we should be running.
        is_limited_mem  = experiment_arguments[2]
        training_loop   = experiment_arguments[5]

        # If the dataset is KITTI*, Cityscapes*, or BDD100K, set the config location.
        dataset_name = experiment_arguments[0]
        if dataset_name.startswith("KITTI"):
            obj_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_kitti_cocofmt.py"
        elif dataset_name.startswith("Cityscapes"):
            obj_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_cityscapes_cocofmt.py"
        elif dataset_name.startswith("BDD100K"):
            obj_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_bdd100k_cocofmt.py"

        if is_limited_mem == 1:
            if training_loop == "obj_det_train":
                from streamline.experiments import LimitedMemoryDetectionExperiment
                experiment_to_run = LimitedMemoryDetectionExperiment(worker_lock, worker_sem, worker_side_pipe, gpu_to_assign, base_dataset_directory, base_exp_directory, db_loc, obj_config_path)
            else:
                from streamline.experiments import LimitedMemoryExperiment
                experiment_to_run = LimitedMemoryExperiment(worker_lock, worker_sem, worker_side_pipe, gpu_to_assign, base_dataset_directory, base_exp_directory, db_loc)           
        elif is_limited_mem == 0:
            if training_loop == "obj_det_train":
                from streamline.experiments import UnlimitedMemoryDetectionExperiment
                experiment_to_run = UnlimitedMemoryDetectionExperiment(worker_lock, worker_sem, worker_side_pipe, gpu_to_assign, base_dataset_directory, base_exp_directory, db_loc, obj_config_path)
            else:
                from streamline.experiments import UnlimitedMemoryExperiment
                experiment_to_run = UnlimitedMemoryExperiment(worker_lock, worker_sem, worker_side_pipe, gpu_to_assign, base_dataset_directory, base_exp_directory, db_loc) 

        # Create and run a worker process
        worker_process = mp.Process(target=_worker_entry, args=(experiment_to_run, experiment_arguments, gpu_to_assign, gpu_availability_queue, distil_loc))
        worker_process.start()
        all_spawned_processes.append(worker_process)

    # Wait for all worker processes to finish
    for worker_process in all_spawned_processes:
        worker_process.join()