from distil.active_learning_strategies.strategy import Strategy

import numpy as np

class ReservoirWrapperStrategy(Strategy):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(ReservoirWrapperStrategy, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)

        # See if there is a buffer capacity. This is needed for reservoir selection!
        if "buffer_capacity" not in args:
            raise ValueError("No buffer capacity present in 'args' argument!")
        else:
            self.buffer_capacity = args["buffer_capacity"]

        self.num_tasks = len(labeled_dataset.task_idx_partitions)

        # See if there is an initial reservoir counter. This will allow to pick up from where an experiment left off.
        # Otherwise, the reservoir counter is simply initialized as the buffer capacity + 1.
        if "reservoir_counter" not in args:
            self.reservoir_counter = self.buffer_capacity + 1
        else:
            self.reservoir_counter = args["reservoir_counter"]

        # See if there is a wrapped al strategy in args. If so, create the wrapped strategy.
        if "wrapped_al_strategy" not in args:
            raise ValueError("No wrapped al strategy present in 'args' argument!")
        else:
            self.wrapped_al_strategy = args["wrapped_al_strategy"](labeled_dataset, unlabeled_dataset, net, nclasses, args)


    def reservoir(self, selected_unlabeled_idx):

        # To perform reservoir sampling, we shall create a temporary "buffer" of indices that map to the
        # labeled and unlabeled portions (idx, is_labeled). When we do the reservoir update, an index that falls within 
        # the buffer range will simply have its mapping updated.
        temp_buffer_idx_map = {}
        for to_labeled_idx in range(len(self.labeled_dataset)):
            temp_buffer_idx_map[to_labeled_idx] = (to_labeled_idx, True)
        
        # Do the reservoir update for every potential unlabeled idx to add
        for to_unlabeled_idx in selected_unlabeled_idx:

            # Calculated the number of samples in the replay buffer, which are those that are kept in both sets.
            current_samples_in_buffer = len(temp_buffer_idx_map.keys())

            # If the buffer is not at capacity, go ahead and add the instance.
            if current_samples_in_buffer < self.buffer_capacity:
                temp_buffer_idx_map[current_samples_in_buffer] = (to_unlabeled_idx, False)
            else:
                # Since the buffer IS at capacity, generate a random integer between 0 and the reservoir 
                # counter.
                replace_index = np.random.randint(0,self.reservoir_counter)
                
                # If that integer ends up being within the valid range of the buffer, then replace the element
                # at that index.
                if replace_index < self.buffer_capacity:   
                    temp_buffer_idx_map[replace_index] = (to_unlabeled_idx, False)
 
                # Increment the reservoir counter
                self.reservoir_counter += 1

        # The reservoir update now has been done for all incoming unlabeled idx.
        # Now, we can formulate the kept labeled idx and the kept unlabeled idx.
        kept_labeled_idx    = [[] for x in range(self.num_tasks)]
        kept_unlabeled_idx  = [[] for x in range(self.num_tasks)]

        for buffer_idx in temp_buffer_idx_map:
            list_index, is_labeled = temp_buffer_idx_map[buffer_idx]
            if is_labeled:
                task_number, _ = self.labeled_dataset.get_task_number_and_index_in_task(list_index)
                kept_labeled_idx[task_number].append(list_index)
            else:
                task_number, _ = self.unlabeled_dataset.get_task_number_and_index_in_task(list_index)
                kept_unlabeled_idx[task_number].append(list_index)

        # We now have the indices with respect to self.labeled_dataset and self.unlabeled_dataset
        # that should be kept in a reservoir buffer. Return both partitions.
        return kept_labeled_idx, kept_unlabeled_idx


    def select(self, budget):

        self.model.eval()

        # Perform the active learning selection as usual.
        selected_unlabeled_idx = self.wrapped_al_strategy.select(budget)

        # Now, figure out which of these will be added to the labeled set via reservoir sampling.
        kept_labeled_idx, kept_unlabeled_idx = self.reservoir(selected_unlabeled_idx)
        return kept_labeled_idx, kept_unlabeled_idx