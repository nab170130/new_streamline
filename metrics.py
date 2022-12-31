from streamline import metrics, persistence
from tqdm import tqdm

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading as th

def _worker_entry(db_loc, metric, assigned_gpu, gpu_availability_queue, print_thread_queue, distil_loc=None):
    
    # Add DISTIL location to path variable if needed
    if distil_loc is not None:
        sys.path.append(distil_loc)

    metric.evaluate()

    persistence.add_metric_time_value_to_run(db_loc, metric.dataset_name, metric.model_architecture_name, metric.limited_mem, metric.arrival_pattern, metric.run_number, metric.training_loop, 
                                             metric.al_method, metric.al_budget, metric.init_task_size, metric.unl_buffer_size, metric.round_num, metric.name, metric.eval_dataset_name, metric.value, metric.time)

    # Add the GPU back to the queue. Put the completed experiment in the
    # print thread queue.
    completed_string = F"{metric.name} ({metric.dataset_name}_{metric.eval_dataset_name}_{metric.run_number}_{metric.round_num})"
    print_thread_queue.put(completed_string)
    gpu_availability_queue.put(assigned_gpu)


def get_metric_tuples_to_compute(db_loc, metric_configurations):

    # Go through each metric configuration and determine which metrics are missing or not present.
    metric_tuples_to_compute = []

    for train_dataset_name, metric_name, eval_dataset_name in metric_configurations:
        missing_or_invalid_metric_tuples = persistence.get_al_rounds_without_valid_eval_dataset_metric(db_loc, train_dataset_name, metric_name, eval_dataset_name)
        metric_tuples_to_compute.append((metric_name, eval_dataset_name, missing_or_invalid_metric_tuples))
    
    return metric_tuples_to_compute


def print_thread_entry(print_thread_queue, total_metric_tuples_to_compute):
    
    with tqdm(total=total_metric_tuples_to_compute) as pbar:
        while True:
            # This statement will block if nothing is yet in the queue.
            completed_metric_string = print_thread_queue.get()

            pbar.set_description(F"Completed: {completed_metric_string}")
            pbar.update(1)
    

if __name__ == "__main__":
    
    # Set start method to spawn(). CUDA cannot be initialized again in forked
    # subprocesses.
    mp.set_start_method("spawn")

    # Retrieve the experiment configuration from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", action=argparse.BooleanOptionalAction, help="Whether to recalculate all metrics in config")
    parser.add_argument("metric_config", type=str, help="Location of config file (json)")
    args = parser.parse_args()
    json_config_loc = args.metric_config
    override = args.override
    with open(json_config_loc, "r") as f:
        metric_config = json.load(f)
    
    # Get DISTIL's location if needed/provided
    if "distil_directory" in metric_config:
        distil_loc = metric_config["distil_directory"]
        sys.path.append(distil_loc)
    else:
        distil_loc = None

    # Get the list of GPUs that should be used.
    # Note that we make the assumption that one
    # process requires one full GPU. GPUs are 
    # assigned based on a simple queueing mechanism.
    gpu_list = metric_config["gpus"]
    n_proc = len(gpu_list)
    gpu_availability_queue = mp.Queue()
    for gpu in gpu_list:
        gpu_availability_queue.put(gpu)

    batch_size = metric_config["batch_size"]
    
    # Get saving locations
    base_dataset_directory = metric_config["dataset_directory"]
    base_exp_directory = metric_config["base_exp_directory"]
    db_loc = metric_config["db_loc"]
    
    # Get all metrics that need to be computed. Additionally, record all spawned processes
    metric_tuples_to_compute = get_metric_tuples_to_compute(db_loc, metric_config["metrics"])
    total_metric_tuples_to_compute = sum([len(x) for _, _, x in metric_tuples_to_compute])

    print(F"Computing {total_metric_tuples_to_compute} metrics")

    # Start the print thread
    print_thread_queue = mp.Queue()
    parent_print_thread = th.Thread(target=print_thread_entry, args=(print_thread_queue, total_metric_tuples_to_compute), daemon=True)
    parent_print_thread.start() 

    all_spawned_processes = []
    for metric_name, eval_dataset_name, metric_tuple_list in metric_tuples_to_compute:
        for round_join_metric_tuple in metric_tuple_list:

            # Stop if there are no GPUs available. If so, wait until one of the child processes finishes.
            # While get() blocks below if the queue is empty, it is preferred to block until the child
            # process terminates, which will ensure that a new process can initialize CUDA on the opened GPU.
            if gpu_availability_queue.empty():
                os.waitpid(-1, 0)
            
            # Pick a GPU out of the queue
            gpu_to_assign = gpu_availability_queue.get()

            # Update the round x metric tuple to have the right eval dataset name.
            round_join_metric_list = list(round_join_metric_tuple)
            round_join_metric_list[-3] = eval_dataset_name

            if eval_dataset_name.startswith("KITTI"):
                obj_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_kitti_cocofmt.py"
            elif eval_dataset_name.startswith("Cityscapes"):
                obj_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_cityscapes_cocofmt.py"
            elif eval_dataset_name.startswith("BDD100K"):
                obj_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_bdd100k_cocofmt.py"
            else:
                obj_config_path = None

            metric_factory = metrics.MetricFactory(db_loc, base_exp_directory, base_dataset_directory, gpu_to_assign, batch_size, round_join_metric_list, obj_config_path)
            metric = metric_factory.get_metric(metric_name)
        
            # Create and run a worker process
            worker_process = mp.Process(target=_worker_entry, args=(db_loc, metric, gpu_to_assign, gpu_availability_queue, print_thread_queue, distil_loc))
            worker_process.start()
            all_spawned_processes.append(worker_process)

    # Wait for all worker processes to finish
    for worker_process in all_spawned_processes:
        worker_process.join()