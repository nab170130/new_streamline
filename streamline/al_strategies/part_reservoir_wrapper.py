from distil.active_learning_strategies.strategy import Strategy

import copy
import numpy as np

class PartitionReservoirWrapperStrategy(Strategy):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(PartitionReservoirWrapperStrategy, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)

        # See if there is a buffer capacity. This is needed for reservoir selection!
        if "buffer_capacity" not in args:
            raise ValueError("No buffer capacity present in 'args' argument!")
        else:
            self.buffer_capacity = args["buffer_capacity"]

        self.num_tasks = len(labeled_dataset.task_idx_partitions)

        # See if there is an initial reservoir counter. This will allow to pick up from where an experiment left off.
        # Otherwise, the reservoir counter is simply initialized as the buffer capacity + 1.
        if "reservoir_counters" not in args:
            self.reservoir_counters = [(self.buffer_capacity // self.num_tasks) + 1 for x in range(self.num_tasks)]
        else:
            self.reservoir_counters = args["reservoir_counters"]

        # See if there is a wrapped al strategy in args. If so, create the wrapped strategy.
        if "wrapped_al_strategy" not in args:
            raise ValueError("No wrapped al strategy present in 'args' argument!")
        else:
            self.wrapped_al_strategy = args["wrapped_al_strategy"](labeled_dataset, unlabeled_dataset, net, nclasses, args)


    def get_oracle_task_identity(self):

        # We can get the task identity by using the get_task_number_and_index_in_task() method on any of the idx in the
        # unlabeled dataset. This is because all the incoming data in the unlabeled dataset belongs to one task.
        task_number, _ = self.unlabeled_dataset.get_task_number_and_index_in_task(0)
        return task_number


    def part_reservoir(self, selected_unlabeled_idx, task_identity):

        # The notion here is similar to regular reservoir sampling; however, we assume that there are self.num_tasks partitions
        # of the memory budget, admitting self.num_tasks reservoirs. Hence, part_reservoir only inserts instances into the specified
        # reservoir partition.

        # To perform reservoir sampling, we shall create a temporary "buffer" of indices that map to the
        # labeled and unlabeled portions (idx, is_labeled). When we do the reservoir update, an index that falls within 
        # the buffer range will simply have its mapping updated. Here, we add only the indices of the identified partition.
        temp_buffer_idx_map = {}

        in_labeled_dataset_start_partition_idx = sum([len(partition) for partition in self.labeled_dataset.task_idx_partitions[:task_identity]])
        in_labeled_dataset_end_partition_idx = in_labeled_dataset_start_partition_idx + len(self.labeled_dataset.task_idx_partitions[task_identity])
        for in_partition_idx, labeled_dataset_idx in enumerate(range(in_labeled_dataset_start_partition_idx, in_labeled_dataset_end_partition_idx)):
            temp_buffer_idx_map[in_partition_idx] = (labeled_dataset_idx, True)
        
        # Do the reservoir update for every potential unlabeled idx to add
        for to_unlabeled_idx in selected_unlabeled_idx:

            # Calculated the number of samples in the replay buffer, which are those that are kept in both sets.
            current_samples_in_buffer = len(temp_buffer_idx_map.keys())

            # If the buffer partition is not at capacity, go ahead and add the instance.
            if current_samples_in_buffer < self.buffer_capacity // self.num_tasks:
                temp_buffer_idx_map[current_samples_in_buffer] = (to_unlabeled_idx, False)
            else:
                # Since the buffer partition IS at capacity, generate a random integer between 0 and the reservoir 
                # counter corresponding to the identified partition.
                replace_index = np.random.randint(0,self.reservoir_counters[task_identity])
                
                # If that integer ends up being within the valid range of the buffer partition, then replace the element
                # at that index.
                if replace_index < self.buffer_capacity // self.num_tasks:   
                    temp_buffer_idx_map[replace_index] = (to_unlabeled_idx, False)
 
                # Increment the reservoir counter
                self.reservoir_counters[task_identity] += 1

        # The reservoir update now has been done for all incoming unlabeled idx.
        # Now, we can formulate the kept labeled idx and the kept unlabeled idx.
        # Again, there is a slight difference from the no partitioning version. We keep all 
        # the rest of the indices of the other partitions while mapping this partition back.
        kept_labeled_idx    = [[] for x in range(self.num_tasks)]
        kept_unlabeled_idx  = [[] for x in range(self.num_tasks)]

        for buffer_idx in temp_buffer_idx_map:
            list_index, is_labeled = temp_buffer_idx_map[buffer_idx]
            if is_labeled:
                kept_labeled_idx[task_identity].append(list_index)
            else:
                kept_unlabeled_idx[task_identity].append(list_index)

        # Add the rest of the partitions back as previously mentioned
        in_labeled_dataset_start_partition_idx = 0
        in_labeled_dataset_end_partition_idx = 0
        for task_num, partition in enumerate(self.labeled_dataset.task_idx_partitions):
            in_labeled_dataset_end_partition_idx += len(partition)
            if task_num != task_identity:
                kept_labeled_idx[task_num] = list(range(in_labeled_dataset_start_partition_idx,in_labeled_dataset_end_partition_idx))
            in_labeled_dataset_start_partition_idx = in_labeled_dataset_end_partition_idx

        # We now have the indices with respect to self.labeled_dataset and self.unlabeled_dataset
        # that should be kept in a reservoir buffer. Return both partitions.
        return kept_labeled_idx, kept_unlabeled_idx


    def select(self, budget):

        self.model.eval()

        # Perform the active learning selection as usual.
        selected_unlabeled_idx = self.wrapped_al_strategy.select(budget)

        # Now, figure out which of these will be added to the labeled set via reservoir sampling.
        task_identity = self.get_oracle_task_identity()
        kept_labeled_idx, kept_unlabeled_idx = self.part_reservoir(selected_unlabeled_idx, task_identity)
        return kept_labeled_idx, kept_unlabeled_idx