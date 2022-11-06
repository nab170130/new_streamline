from .streamline_base import StreamlineBase

import submodlib

class UnlimitedMemoryStreamline(StreamlineBase):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(UnlimitedMemoryStreamline, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
    

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


    def scg_select(self, data_sijs, data_private_sijs, private_private_sijs, budget):

        #Get hyperparameters from args dict
        optimizer = self.args['optimizer'] if 'optimizer' in self.args else 'NaiveGreedy'
        nu = self.args['nu'] if 'nu' in self.args else 1
        stopIfZeroGain = self.args['stopIfZeroGain'] if 'stopIfZeroGain' in self.args else False
        stopIfNegativeGain = self.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.args else False
        verbose = self.args['verbose'] if 'verbose' in self.args else False

        # Map SMI function to its SCG variant and form its conditional gain function:
        #   1. fl1mi    -> flcg
        #   2. gcmi     -> fccg
        #   3. logdetmi -> logdetcg
        if(self.args['obj_function']=='flcg'):
            obj = submodlib.FacilityLocationConditionalGainFunction(n=data_sijs.shape[0],
                                                                      num_privates=private_private_sijs.shape[0],  
                                                                      data_sijs=data_sijs, 
                                                                      private_sijs=data_private_sijs, 
                                                                      privacyHardness=nu)
        
        if(self.args['obj_function']=='gccg'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.GraphCutConditionalGainFunction(n=data_sijs.shape[0],
                                                                      num_privates=private_private_sijs.shape[0],
                                                                      lambdaVal=lambdaVal,  
                                                                      data_sijs=data_sijs, 
                                                                      private_sijs=data_private_sijs, 
                                                                      privacyHardness=nu)
        
        if(self.args['obj_function']=='logdetcg'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.LogDeterminantConditionalGainFunction(n=data_sijs.shape[0],
                                                                      num_privates=private_private_sijs.shape[0],
                                                                      lambdaVal=lambdaVal,  
                                                                      data_sijs=data_sijs, 
                                                                      private_sijs=data_private_sijs,
                                                                      private_private_sijs=private_private_sijs, 
                                                                      privacyHardness=nu)

        greedyList = obj.maximize(budget=budget, optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
                              stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
        greedyIndices = [x[0] for x in greedyList]

        return greedyIndices


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
            oversample_budget       = max(min(self.args['acc_budget'], avg_task_size_diff - budget), 0)
            fair_adjusted_budget    = budget + oversample_budget
            self.args['acc_budget'] = self.args['acc_budget'] - oversample_budget
        else:
            size_of_smallest_task   = min([len(partition) for partition in self.labeled_dataset.task_idx_partitions])
            size_of_current_task    = len(self.labeled_dataset.task_idx_partitions[task_identity])
            fair_adjusted_budget    = int(min_budget_factor * budget + (1 - min_budget_factor) * budget * (size_of_smallest_task / size_of_current_task))
            leftover_budget         = budget - fair_adjusted_budget
            self.args['acc_budget'] = self.args['acc_budget'] + leftover_budget

        # Select new unlabeled indices to add. Rearrange these indices to match the identified task.
        selected_unlabeled_idx = self.scg_select(data_sijs, data_private_sijs, private_private_sijs, fair_adjusted_budget)
        selected_unlabeled_idx_partitioned = [[] for x in range(self.num_tasks)]
        selected_unlabeled_idx_partitioned[task_identity].extend(selected_unlabeled_idx)

        return selected_unlabeled_idx_partitioned