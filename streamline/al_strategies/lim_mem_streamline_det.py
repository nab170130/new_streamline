from .streamline_base_det import StreamlineBaseDetection
from ..utils.lazy_greedy_matroid_opt import LazyGreedyMatroidPartitionOptimizer

import submodlib
import torch

class LimitedMemoryStreamlineDetection(StreamlineBaseDetection):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(LimitedMemoryStreamlineDetection, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)


    def calculate_subkernel(self, full_sijs, task_identity):

        # Create the subkernel corresponding to the kernel that would be created using self.unlabeled_dataset and self.labeled_dataset.task_idx_
        # partitions[task_identity].
        start_unlabeled_idx         = 0
        end_unlabeled_idx           = len(self.unlabeled_dataset)
        start_identified_task_idx   = end_unlabeled_idx + sum([len(partition) for partition in self.labeled_dataset.task_idx_partitions[:task_identity]])
        end_identified_task_idx     = start_identified_task_idx + len(self.labeled_dataset.task_idx_partitions[task_identity])
        subset_matrix_idx = list(range(start_unlabeled_idx, end_unlabeled_idx))
        subset_matrix_idx.extend(range(start_identified_task_idx, end_identified_task_idx))
        obj_sijs = full_sijs[subset_matrix_idx][:,subset_matrix_idx]

        return obj_sijs


    def get_optimizer(self, obj_sijs):

        # Extract some information for submodlib
        num_unlabeled_instances = len(self.unlabeled_dataset)
        metric = self.args['metric'] if 'metric' in self.args else 'cosine'

        # Get the regular submodular function from each SMI variant.
        if self.args['obj_function']=='fl':
            obj_func = submodlib.FacilityLocationFunction(n=obj_sijs.shape[0],
                                                                mode="dense",
                                                                separate_rep=False,
                                                                sijs=obj_sijs,
                                                                metric=metric)
    
        elif self.args['obj_function']=='gc':
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 0.5
            obj_func = submodlib.GraphCutFunction(n=obj_sijs.shape[0],
                                                        mode="dense",
                                                        lambdaVal=lambdaVal,
                                                        sijs=obj_sijs,
                                                        metric=metric)
            
        elif self.args['obj_function']=='logdet':
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj_func = submodlib.LogDeterminantFunction(n=obj_sijs.shape[0],
                                                            mode="dense",
                                                            separate_rep=False,
                                                            lambdaVal=lambdaVal,
                                                            sijs=obj_sijs,
                                                            metric=metric)

        unlabeled_partition = set(range(num_unlabeled_instances))
        labeled_partition = set(range(num_unlabeled_instances, obj_sijs.shape[0]))
        ground_set_partitions = [unlabeled_partition, labeled_partition]

        optimizer = LazyGreedyMatroidPartitionOptimizer(obj_func, ground_set_partitions)
        return optimizer


    def select(self, budget):

        self.model.eval()
        
        # Get the semantic segmentation similarity kernel, which will be used for task identification,
        sem_seg_unlab_feat  = self.compute_sem_seg_features(self.unlabeled_dataset)
        sem_seg_lab_feat    = self.compute_sem_seg_features(self.labeled_dataset)
        sem_seg_all_feat    = torch.cat([sem_seg_unlab_feat, sem_seg_lab_feat])
        sem_seg_kern        = self.compute_sem_seg_similarity_kernel(sem_seg_all_feat)
        task_identity       = self.identify_task(sem_seg_kern.cpu().numpy())

        # Now, get the object detection similarity kernel, which will be used for selecting new instances
        obj_det_unlab_feat  = self.compute_obj_det_features(self.unlabeled_dataset, gt_proposals=False)     # obj_det_unlab_feat size: [N_unl, k, d]
        obj_det_lab_feat    = self.compute_obj_det_features(self.labeled_dataset, gt_proposals=True)        # size: [N_lab, k, d]
        obj_det_all_feat    = torch.cat([obj_det_unlab_feat, obj_det_lab_feat])                             # size: [N_unl + N_lab, k, d]
        obj_det_kern        = self.compute_obj_det_similarity_kernel(obj_det_all_feat)                      # size: [N_unl + N_lab, N_unl + N_lab]
        obj_sijs            = self.calculate_subkernel(obj_det_kern, task_identity)                         # size: [N_unl + N_lab.identified, N_unl + N_lab.identified]
        optimizer           = self.get_optimizer(obj_sijs.cpu().numpy())                                    # S_ij: 

        # Constraints are formed as follows:
        #   1. The number of unlabeled points cannot exceed the budget, so the unlabeled portion has a partition constraint of `budget`.
        #   2. As many labeled instances as needed can be selected, so the labeled portion has a partition constraint of `len(lbl_dset[task])`.
        #   3. In sum, no more than the buffer size / num tasks can be selected.
        unlabeled_portion_constraint    = budget
        labeled_portion_constraint      = len(self.labeled_dataset.task_idx_partitions[task_identity])
        partition_constraints           = [unlabeled_portion_constraint, labeled_portion_constraint]
        cardinality_constraint          = len(self.labeled_dataset) // self.num_tasks

        # Use the optimizer to select ground set indices
        selected_ground_set_idx = optimizer.maximize(partition_constraints, cardinality_constraint)

        # We can now formulate the kept portions of what was selected. There are two parts to consider:
        #   1. The rest of the labeled instances belonging to other tasks are kept
        #   2. The selected instances from the [unlabeled, labeled_identified_task] ground set must be mapped back to kep labeled/unlabeled idx.
        kept_labeled_idx = [[] for x in range(self.num_tasks)]
        kept_unlabeled_idx = [[] for x in range(self.num_tasks)]

        # Do step 1, ignoring the identified task.
        start_task_idx = 0
        for task_number in range(self.num_tasks):
            end_task_idx = start_task_idx + len(self.labeled_dataset.task_idx_partitions[task_number])
            if task_number != task_identity:
                kept_labeled_idx[task_number] = list(range(start_task_idx, end_task_idx))
            start_task_idx = end_task_idx 

        # Do step 2, mapping the selected indices back to their labeled and unlabeled idx.
        for to_map_idx in selected_ground_set_idx:
            if to_map_idx < len(self.unlabeled_dataset):

                # The ground set index corresponds to an index in the unlabeled dataset. We do not need
                # to do any additional steps as the index maps directly to the unlabeled dataset.
                kept_unlabeled_idx[task_identity].append(to_map_idx)
            
            else:

                # The ground set index corresponds to an index in the labeled dataset. To map to the 
                # labeled index, we can do the following:
                #   1. Subtract the length of the unlabeled dataset, getting the index wrpt the labeled partition to which it belongs
                #   2. Add the starting index of the partition to the result of step 1, getting the index wrpt to self.labeled_dataset
                in_partition_idx            = to_map_idx - len(self.unlabeled_dataset)
                start_identified_task_idx   = sum([len(partition) for partition in self.labeled_dataset.task_idx_partitions[:task_identity]])
                labeled_idx                 = in_partition_idx + start_identified_task_idx
                kept_labeled_idx[task_identity].append(labeled_idx)

        return kept_labeled_idx, kept_unlabeled_idx