from abc import ABC, abstractmethod

from distil.active_learning_strategies.strategy import Strategy
from distil.utils.utils import LabeledToUnlabeledDataset

from ..utils.lazy_greedy_matroid_opt import LazyGreedyMatroidPartitionOptimizer

import submodlib
import torch

class StreamlineBase(Strategy, ABC):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(StreamlineBase, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        self.num_tasks = len(labeled_dataset.task_idx_partitions)

        if "min_budget" not in args:
            self.args["min_budget"] = 0.5
    

    def identify_task(self, full_sijs, metric='rbf'):

        # If we should be finding oracle task identity, do so and return instead of this.
        if self.args["oracle_task_identity"]:
            task_identity, _ = self.unlabeled_dataset.get_task_number_and_index_in_task(0)
            return task_identity

        # Extract some information for submodlib and for taking subkernels
        num_unlabeled_instances = len(self.unlabeled_dataset)
        eta = self.args['eta'] if 'eta' in self.args else 1

        # The first such view: Take only the pairwise similarities of those points in the unlabeled dataset
        data_sijs = full_sijs[:num_unlabeled_instances,:num_unlabeled_instances]

        # In a loop, calculate the SMI between each coreset and the unlabeled dataset.
        smi_base_fractions = []

        current_coreset_start_range = num_unlabeled_instances
        for task_idx_partition in self.labeled_dataset.task_idx_partitions:

            # The other views need to be calculated specifically for their coreset.
            # We first calculate the ranges of the kernel that we need to slice.
            current_coreset_end_range = current_coreset_start_range + len(task_idx_partition)
            query_sijs = full_sijs[:num_unlabeled_instances,current_coreset_start_range:current_coreset_end_range]
            query_query_sijs = full_sijs[current_coreset_start_range:current_coreset_end_range,current_coreset_start_range:current_coreset_end_range]

            base_indices = list(range(num_unlabeled_instances))
            base_indices.extend(range(current_coreset_start_range, current_coreset_end_range))
            base_sijs = full_sijs[base_indices,:][:, base_indices]

            if(self.args['smi_function']=='fl2mi'):
                smi_obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=num_unlabeled_instances,
                                                                      num_queries=len(task_idx_partition), 
                                                                      query_sijs=query_sijs, 
                                                                      queryDiversityEta=eta)
                base_obj = submodlib.FacilityLocationFunction(n=base_sijs.shape[0],
                                                                mode="dense",
                                                                separate_rep=False,
                                                                sijs=base_sijs,
                                                                metric=metric)
                norm_factor = query_query_sijs.shape[0] + data_sijs.shape[0]
            elif(self.args['smi_function']=='gcmi'):
                lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 0.5
                smi_obj = submodlib.GraphCutMutualInformationFunction(n=num_unlabeled_instances,
                                                                      num_queries=len(task_idx_partition),
                                                                      query_sijs=query_sijs, 
                                                                      metric=metric)
                base_obj = submodlib.GraphCutFunction(n=base_sijs.shape[0],
                                                            mode="dense",
                                                            lambdaVal=lambdaVal,
                                                            ggsijs=base_sijs,
                                                            metric=metric)
                norm_factor = query_query_sijs.shape[0] * data_sijs.shape[0]
            elif(self.args['smi_function']=='logdetmi'):
                lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
                smi_obj = submodlib.LogDeterminantMutualInformationFunction(n=num_unlabeled_instances,
                                                                    num_queries=len(task_idx_partition),
                                                                    data_sijs=data_sijs,  
                                                                    query_sijs=query_sijs,
                                                                    query_query_sijs=query_query_sijs,
                                                                    magnificationEta=eta,
                                                                    lambdaVal=lambdaVal)
                base_obj = submodlib.LogDeterminantFunction(n=base_sijs.shape[0],
                                                                mode="dense",
                                                                lambdaVal=lambdaVal,
                                                                sijs=base_sijs,
                                                                metric=metric)
                norm_factor = 1

            # Evaluate the smi objective function and the base objective function to get the fraction between the two.
            # Note that the submodular mutual information between two sets is always less than the base objective value
            # of each set (for monotonic functions).
            submodular_mutual_information_objective_value   = smi_obj.evaluate(set(range(len(self.unlabeled_dataset))))
            base_objective_value                            = base_obj.evaluate(set(range(num_unlabeled_instances, num_unlabeled_instances + len(task_idx_partition))))

            print(F"FRAC {submodular_mutual_information_objective_value / norm_factor} ACTUAL TASK {self.unlabeled_dataset.get_task_number_and_index_in_task(0)[0]}")

            # Update the range for the next iteration
            current_coreset_start_range = current_coreset_end_range

            # Store the objective values
            smi_base_fractions.append(submodular_mutual_information_objective_value / norm_factor)#base_objective_value)
        
        self.smi_base_fractions = smi_base_fractions

        # Determine which SMI-base fraction was the highest. We predict that the coreset with the highest fraction gives the task identity
        task_identity = None
        max_task_fraction = -float("inf")
        for task_idx, smi_base_fraction in enumerate(smi_base_fractions):
            if smi_base_fraction > max_task_fraction:
                max_task_fraction = smi_base_fraction
                task_identity = task_idx

        print("I choose", task_identity, max_task_fraction)

        return task_identity


    def calculate_kernel(self, metric='rbf'):

        # Get hyperparameters from args dict
        gradType = self.args['gradType'] if 'gradType' in self.args else "bias_linear"
        embedding_type = self.args['embedding_type'] if 'embedding_type' in self.args else "features"
        if(embedding_type=="features"):
            layer_name = self.args['layer_name'] if 'layer_name' in self.args else "avgpool"

        # Compute the embeddings of the unlabeled dataset
        if embedding_type == "gradients":
            unlabeled_data_embedding = self.get_grad_embedding(self.unlabeled_dataset, True, gradType)
        elif embedding_type == "features":
            unlabeled_data_embedding = self.get_embedding(self.unlabeled_dataset)
        else:
            raise ValueError("Provided representation must be one of gradients or features")
        
        # Compute the embeddings of the labeled dataset, which is the union of all the query sets we will be 
        # using for SMI.
        if embedding_type == "gradients":
            coreset_data_embedding = self.get_grad_embedding(self.labeled_dataset, False, gradType)
        elif embedding_type == "features":
            coreset_data_embedding = self.get_embedding(LabeledToUnlabeledDataset(self.labeled_dataset))
        else:
            raise ValueError("Provided representation must be one of gradients or features")

        # Compute the full kernel. We will take views of this kernel for the other kernels that are used in the computation
        full_embedding = torch.cat([unlabeled_data_embedding, coreset_data_embedding])

        if metric == "rbf":
            
            # Generate pairwise distances based on the embedding type. Here, we also attempt to normalize the distances to some degree
            # so that most similarity values in the kernel do not vanish (e^-20, for example, may as well be 0).
            pairwise_distances      = torch.cdist(full_embedding.to(self.device), full_embedding.to(self.device))
            rbf_sig                 = torch.max(pairwise_distances) / 2     # Div by 4 so that the max distance has z-score of 4.
            pw_square_distances     = torch.pow(pairwise_distances, 2)
            norm_pw_sq_distances    = pw_square_distances / (2. * rbf_sig * rbf_sig)
            full_sijs = rbf_kern    = torch.exp(-norm_pw_sq_distances).cpu().numpy()

        else:

            full_sijs = (submodlib.helper.create_kernel(X=full_embedding.cpu().numpy(), metric=metric, method="sklearn") + 1.) / 2.

        return full_sijs


    @abstractmethod
    def select(self, budget):
        pass