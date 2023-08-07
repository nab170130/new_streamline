from abc import ABC, abstractmethod

from ..utils.lazy_greedy_matroid_opt import LazyGreedyMatroidPartitionOptimizer
from .streamline_base_det import StreamlineBaseDetection

from mmcv import Config
from mmseg.apis import init_segmentor
from mmdet.core import bbox2roi
from mmseg.datasets import build_dataloader

import submodlib
import torch
from tqdm import tqdm

class Talisman(StrategyBaseDetection):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(Talisman, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        self.num_tasks          = len(labeled_dataset.task_idx_partitions)
        self.cfg                = args["cfg"]
        self.max_prop_obj_det   = 50


    def smi_select(self, data_sijs, query_sijs, query_query_sijs, budget):

        #Get hyperparameters from args dict
        optimizer = self.args['optimizer'] if 'optimizer' in self.args else 'LazyGreedy'
        eta = self.args['eta'] if 'eta' in self.args else 1
        stopIfZeroGain = self.args['stopIfZeroGain'] if 'stopIfZeroGain' in self.args else False
        stopIfNegativeGain = self.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.args else False
        verbose = self.args['verbose'] if 'verbose' in self.args else False

        if(self.args['smi_function']=='fl1mi'):
            obj = submodlib.FacilityLocationMutualInformationFunction(n=data_sijs.shape[0],
                                                                      num_queries=query_query_sijs.shape[0], 
                                                                      data_sijs=data_sijs , 
                                                                      query_sijs=query_sijs, 
                                                                      magnificationEta=eta)

        if(self.args['smi_function']=='fl2mi'):
            obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=data_sijs.shape[0],
                                                                      num_queries=query_query_sijs.shape[0],
                                                                      query_sijs=query_sijs, 
                                                                      queryDiversityEta=eta)
        
        if(self.args['smi_function']=='com'):
            obj = submodlib.ConcaveOverModularFunction(n=data_sijs.shape[0],
                                                                      num_queries=query_query_sijs.shape[0],
                                                                      query_sijs=query_sijs, 
                                                                      queryDiversityEta=eta)
        if(self.args['smi_function']=='gcmi'):
            obj = submodlib.GraphCutMutualInformationFunction(n=data_sijs.shape[0],
                                                                      num_queries=query_query_sijs.shape[0],
                                                                      query_sijs=query_sijs, 
                                                                      metric=metric)
        if(self.args['smi_function']=='logdetmi'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.LogDeterminantMutualInformationFunction(n=data_sijs.shape[0],
                                                                    num_queries=query_query_sijs.shape[0],
                                                                    data_sijs=data_sijs,  
                                                                    query_sijs=query_sijs,
                                                                    query_query_sijs=query_query_sijs,
                                                                    magnificationEta=eta,
                                                                    lambdaVal=lambdaVal)

        greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
                              stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
        greedyIndices = [x[0] for x in greedyList]
        return greedyIndices


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


    def compute_obj_det_similarity_kernel(self, n_k_d_features):

        print("Computing object detection similarity kernel")

        n, k, d = n_k_d_features.shape

        # Normalize each of the N * k feature vectors. To do so, we compute each's norm, broadcast the N * k norms to a 
        # compatible shape, and do elementwise division. Zero-vectors have zero norm, so to avoid division errors, we set
        # such norms to 1 (which leaves the vector unchanged)
        per_feature_vector_norm                             = torch.linalg.norm(n_k_d_features, dim=-1)[:,:,None]
        per_feature_vector_norm[per_feature_vector_norm==0] = 1.
        normalized_feature_tensor                           = n_k_d_features / per_feature_vector_norm

        # For averaging purposes, compute an n tensor that counts which of the n x k vectors (d dim) are nonzero vectors. Since some of the k
        # proposals for each image may be zero vectors, we do not want them to contribute to the average. As a corner case, if ALL proposals
        # are zero vectors, then we can set the recorded count to 1 since the sum component of the average will still be zero when averaging
        # the object-to-object similarities.
        n_k_is_nonzero_vector                                                            = torch.any(torch.ne(normalized_feature_tensor, 0.), dim=2)
        image_proposal_nonzero_vector_counts                                             = torch.sum(1.0 * n_k_is_nonzero_vector, dim=1)
        image_proposal_nonzero_vector_counts[image_proposal_nonzero_vector_counts == 0.] = 1.

        # Now, we can begin computing similarities for each pair of n images. The similarity is computed
        # by taking the average cosine similarity between the k^2 pairs of proposals (which is why we 
        # normalized the feature vectors). To avoid creating a monolithic N x N x k x k tensor, we compute
        # submatrices of the N x N target matrix in batches.
        start_idx = 0
        image_image_similarity_kernel = torch.zeros(n,n).to(n_k_d_features.device)

        with tqdm(total=n) as pbar:
            while start_idx != n:
                
                # Get the batch of images that correspond to this iteration
                end_idx                                 = min(start_idx + self.args["batch_size"], n)
                batch_normalized_features               = normalized_feature_tensor[start_idx:end_idx,:,:]
                batch_proposal_nonzero_vector_counts    = image_proposal_nonzero_vector_counts[start_idx:end_idx]
                nbatch                                  = batch_normalized_features.shape[0]

                # Compute the cosine similarities between each of the k^2 proposals between each pair
                # of images between the batch and the whole feature tensor. This is done by contracting
                # along the last dimension (the d-dimensional feature vectors). 
                # 
                # Next, the similarity of an image-image pair is computed by taking a max along the object similarities.
                nbatch_n_k_k_kern                                   = torch.tensordot(batch_normalized_features, normalized_feature_tensor, dims=([-1],[-1])).permute(0,2,1,3)
                nbatch_n_max                                        = torch.amax(nbatch_n_k_k_kern, dim=(2,3))
                image_image_similarity_kernel[start_idx:end_idx]    = nbatch_n_avg

                # Update the new start idx for the next iteration
                pbar.update(end_idx - start_idx)
                start_idx = end_idx

        # Lastly, since the values of the computed kernel could be between -1 and 1, do a simple fix to ensure those ranges are between 0 and 1.
        non_negative_image_image_similarity_kernel = (image_image_similarity_kernel + 1.) / 2.
        return non_negative_image_image_similarity_kernel


    def select(self, budget):
       
        self.model.eval()

        self.args['smi_function']   = "fl2mi"
        task_identity               = self.num_tasks - 1

        # Now, get the object detection similarity kernel, which will be used for selecting new instances
        obj_det_unlab_feat                                  = self.compute_obj_det_features(self.unlabeled_dataset, gt_proposals=False)
        obj_det_lab_feat                                    = self.compute_obj_det_features(self.labeled_dataset, gt_proposals=True)
        obj_det_all_feat                                    = torch.cat([obj_det_unlab_feat, obj_det_lab_feat])
        obj_det_kern                                        = self.compute_obj_det_similarity_kernel(obj_det_all_feat)
        data_sijs, data_query_sijs, query_query_sijs        = self.calculate_subkernels(obj_det_kern, task_identity)

        # Select new unlabeled indices to add. Rearrange these indices to match the identified task.
        selected_unlabeled_idx = self.smi_select(data_sijs.cpu().numpy(), data_query_sijs.cpu().numpy(), query_query_sijs.cpu().numpy(), budget)
        selected_unlabeled_idx_partitioned = [[] for x in range(self.num_tasks)]
        selected_unlabeled_idx_partitioned[task_identity].extend(selected_unlabeled_idx)

        return selected_unlabeled_idx_partitioned