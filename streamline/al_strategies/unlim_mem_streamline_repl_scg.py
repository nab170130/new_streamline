from .streamline_base import StreamlineBase
from distil.active_learning_strategies import BADGE

import numpy as np
import submodlib

class UnlimitedMemoryStreamlineReplacedSCG(StreamlineBase):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(UnlimitedMemoryStreamlineReplacedSCG, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
    

    def calculate_subkernels(self, full_sijs, task_identity):

        # Create the subkernel corresponding to the kernel that would be created using self.unlabeled_dataset and self.labeled_dataset.task_idx_
        # partitions[task_identity].
        start_unlabeled_idx         = 0
        end_unlabeled_idx           = len(self.unlabeled_dataset)
        start_identified_task_idx   = end_unlabeled_idx + sum([len(partition) for partition in self.labeled_dataset.task_idx_partitions[:task_identity]])
        end_identified_task_idx     = start_identified_task_idx + len(self.labeled_dataset.task_idx_partitions[task_identity])
        data_idx = list(range(start_unlabeled_idx, end_unlabeled_idx))
        private_idx = list(range(start_identified_task_idx, end_identified_task_idx))
        
        data_sijs               = full_sijs[data_idx][:,data_idx]
        data_private_sijs       = full_sijs[data_idx][:,private_idx]
        private_private_sijs    = full_sijs[private_idx][:,private_idx]

        return data_sijs, data_private_sijs, private_private_sijs


    def select(self, budget):

        self.model.eval()
        
        # Get the similarity kernel, which will be used for task identification and coreset selection
        # Use the similarity kernel to identify the task
        self.args['embedding_type'] = "features"
        full_sijs                                           = self.calculate_kernel()
        task_identity                                       = self.identify_task(full_sijs)

        full_sijs                                           = self.calculate_kernel()
        data_sijs, data_private_sijs, private_private_sijs  = self.calculate_subkernels(full_sijs, task_identity)

        # IF the task is the rare task, then apply the accumulated budget to the base budget.
        # Otherwise, adjust the budget to be fair to task sizes.
        num_tasks           = len(self.labeled_dataset.task_idx_partitions)
        min_budget_factor   = 0.5
        if task_identity == num_tasks - 1:
            avg_task_size           = sum([len(x) for x in self.labeled_dataset.task_idx_partitions[:num_tasks - 1]]) // (num_tasks - 1)    # Avg size of non-rare tasks
            avg_task_size_diff      = avg_task_size - len(self.labeled_dataset.task_idx_partitions[task_identity])
            oversample_budget       = int(max(min(self.args['acc_budget'], avg_task_size_diff - budget), 0))
            fair_adjusted_budget    = budget + oversample_budget
            self.args['acc_budget'] = self.args['acc_budget'] - oversample_budget
        else:
            size_of_smallest_task   = min([len(partition) for partition in self.labeled_dataset.task_idx_partitions])
            size_of_current_task    = len(self.labeled_dataset.task_idx_partitions[task_identity])
            fair_adjusted_budget    = int(min_budget_factor * budget + (1 - min_budget_factor) * budget * (size_of_smallest_task / size_of_current_task))
            leftover_budget         = budget - fair_adjusted_budget
            self.args['acc_budget'] = self.args['acc_budget'] + leftover_budget

        # Ignore the SCG component and instead select using BADGE. Go ahead and delete full_sijs to make room for BADGE.
        proxy_strategy          = BADGE(self.labeled_dataset, self.unlabeled_dataset, self.model, self.target_classes, self.args)
        selected_unlabeled_idx  = proxy_strategy.select(fair_adjusted_budget)
        selected_unlabeled_idx_partitioned = [[] for x in range(self.num_tasks)]
        selected_unlabeled_idx_partitioned[task_identity].extend(selected_unlabeled_idx)

        return selected_unlabeled_idx_partitioned