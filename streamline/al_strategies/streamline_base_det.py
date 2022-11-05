from abc import ABC, abstractmethod

from ..utils.lazy_greedy_matroid_opt import LazyGreedyMatroidPartitionOptimizer

from distil.active_learning_strategies.strategy import Strategy
from mmcv import Config
from mmseg.apis import init_segmentor
from mmdet.core import bbox2roi
from mmseg.datasets import build_dataloader

import submodlib
import torch
from tqdm import tqdm

class StreamlineBaseDetection(Strategy, ABC):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(StreamlineBaseDetection, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        self.num_tasks          = len(labeled_dataset.task_idx_partitions)
        self.cfg                = args["cfg"]
        self.max_prop_obj_det   = 100
    

    def identify_task(self, full_sijs):

        # If we should be finding oracle task identity, do so and return instead of this.
        if self.args["oracle_task_identity"]:
            task_identity, _ = self.unlabeled_dataset.get_task_number_and_index_in_task(0)
            return task_identity

        # Extract some information for submodlib and for taking subkernels
        num_unlabeled_instances = len(self.unlabeled_dataset)
        eta = self.args['eta'] if 'eta' in self.args else 1
        metric = self.args['metric'] if 'metric' in self.args else 'cosine'

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

            elif(self.args['smi_function']=='fl1mi'):
                smi_obj = submodlib.FacilityLocationMutualInformationFunction(n=num_unlabeled_instances,
                                                                      num_queries=len(task_idx_partition), 
                                                                      data_sijs=data_sijs, 
                                                                      query_sijs=query_sijs, 
                                                                      magnificationEta=eta)
                base_obj = submodlib.FacilityLocationFunction(n=full_sijs.shape[0],
                                                                mode="dense",
                                                                separate_rep=False,
                                                                sijs=full_sijs,
                                                                metric=metric)
        
            elif(self.args['smi_function']=='gcmi'):
                lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 0.5
                smi_obj = submodlib.GraphCutMutualInformationFunction(n=num_unlabeled_instances,
                                                                      num_queries=len(task_idx_partition),
                                                                      query_sijs=query_sijs, 
                                                                      metric=metric)
                base_obj = submodlib.GraphCutFunction(n=full_sijs.shape[0],
                                                            mode="dense",
                                                            lambdaVal=lambdaVal,
                                                            ggsijs=full_sijs,
                                                            metric=metric)
                
            elif(self.args['smi_function']=='logdetmi'):
                lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
                smi_obj = submodlib.LogDeterminantMutualInformationFunction(n=num_unlabeled_instances,
                                                                    num_queries=len(task_idx_partition),
                                                                    data_sijs=data_sijs,  
                                                                    query_sijs=query_sijs,
                                                                    query_query_sijs=query_query_sijs,
                                                                    magnificationEta=eta,
                                                                    lambdaVal=lambdaVal)
                base_obj = submodlib.LogDeterminantFunction(n=full_sijs.shape[0],
                                                                mode="dense",
                                                                lambdaVal=lambdaVal,
                                                                sijs=full_sijs,
                                                                metric=metric)

            # Evaluate the smi objective function and the base objective function to get the fraction between the two.
            # Note that the submodular mutual information between two sets is always less than the base objective value
            # of each set (for monotonic functions).
            submodular_mutual_information_objective_value   = smi_obj.evaluate(set(range(len(self.unlabeled_dataset))))
            base_objective_value                            = base_obj.evaluate(set(range(num_unlabeled_instances, num_unlabeled_instances + len(task_idx_partition))))

            print(F"SUBMOD {submodular_mutual_information_objective_value} BASE {base_objective_value} ACTUAL TASK {self.unlabeled_dataset.get_task_number_and_index_in_task(0)[0]}")

            # Update the range for the next iteration
            current_coreset_start_range = current_coreset_end_range

            # Store the objective values
            smi_base_fractions.append(submodular_mutual_information_objective_value)#base_objective_value)
        
        # Determine which SMI-base fraction was the highest. We predict that the coreset with the highest fraction gives the task identity
        task_identity = None
        max_task_fraction = -float("inf")
        for task_idx, smi_base_fraction in enumerate(smi_base_fractions):
            if smi_base_fraction > max_task_fraction:
                max_task_fraction = smi_base_fraction
                task_identity = task_idx

        print("I choose", task_identity, max_task_fraction)

        return task_identity


    def compute_sem_seg_features(self, dataset):

        dataloader  = build_dataloader(dataset, self.cfg["data"]["samples_per_gpu"], self.cfg["data"]["workers_per_gpu"], 
                                        num_gpus=1, dist=True, shuffle=False, seed=self.cfg["seed"], persistent_workers=False)

        # This method uses a pre-trained semantic segmentor to extract features at the image level.
        # Since segmentors use information about the full image when segmenting, the feature space 
        # induced by the backbone of the segmentor should be well-separated by scene information.
        # First, get the pre-trained segmentor via mmsegmentation
        psp_config_path         = "streamline/utils/mmseg_configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"
        psp_pretrained_path     = "streamline/utils/mmseg_configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"
        model                   = init_segmentor(psp_config_path, psp_pretrained_path, device=self.args["device"])
        model.eval()
        
        # Use the model's backbone (and possibly a neck) to extract features from each image.
        extracted_semantic_seg_features = None
        with torch.no_grad():
            start_idx = 0
            for loaded_batch in dataloader:
                end_idx = start_idx + loaded_batch['img'].data[0].shape[0]

                # Do the model's forward pass util the last prediction layer.
                batch_features = model.backbone(loaded_batch['img'].data[0].to(self.args["device"])) # post-aug collated batch tensor is in ['img'][0]
                batch_features = model.decode_head(batch_features)

                # Batch features returns a 4-tuple of embeddings. We use the last embedding to get features. However, 
                # these feature embeddings are quite large (2048 x 72 x 128), so we opt to reduce these by taking a
                # max pool to reduce the spatial extent and to take an average over each channel to reduce the size
                # to 2304-dimensional embeddings
                max_pool_batch_features                 = torch.nn.functional.max_pool2d(batch_features, kernel_size=16, stride=16, padding=0, dilation=1)
                single_channel_max_pool_batch_features  = torch.flatten(max_pool_batch_features, 1)
                if extracted_semantic_seg_features is None:
                    extracted_semantic_seg_features = torch.zeros(len(dataset), single_channel_max_pool_batch_features.shape[1])
                extracted_semantic_seg_features[start_idx:end_idx] = single_channel_max_pool_batch_features
                start_idx = end_idx

        return extracted_semantic_seg_features

    
    def _get_RoI_features(self, model, features, proposals):

        # Convert proposals to RoIs and compute flattened bounding box features.
        rois = bbox2roi(proposals).to(device=self.args["device"])  
        bbox_feats = model.roi_head.bbox_roi_extractor(features[:model.roi_head.bbox_roi_extractor.num_inputs], rois)
        if model.roi_head.with_shared_head:
            bbox_feats = model.roi_head.shared_head(bbox_feats)
        flattened_bbox_feats = bbox_feats.flatten(1)

        # Compute the FC features and the scores for each class
        fc_features = flattened_bbox_feats
        for fc in model.roi_head.bbox_head.shared_fcs:
            fc_features = model.roi_head.bbox_head.relu(fc(fc_features))
        cls_scores = model.roi_head.bbox_head.fc_cls(fc_features) if model.roi_head.bbox_head.with_cls else None
        
        return fc_features, cls_scores       


    def compute_obj_det_features(self, dataset, gt_proposals=False):

        dataloader  = build_dataloader(dataset, self.cfg["data"]["samples_per_gpu"], self.cfg["data"]["workers_per_gpu"], 
                                        num_gpus=1, dist=True, shuffle=False, seed=self.cfg["seed"], persistent_workers=False)
        model       = self.model.to(self.args["device"])

        # Here, we will be extracting the top-k proposal feature vectors for each image. Here, "top-k"
        # refers to the most confident bbox predictions for a given class.
        extracted_feature_vectors   = None
        start_idx                   = 0
        for loaded_batch in dataloader:

            # Get the feature vectors at the feature-map level of the model pipeline. Get features throughout the forward process.
            loaded_batch_img        = loaded_batch['img'].data[0].to(self.args["device"]) # post-aug collated batch tensor is in ['img'][0]
            loaded_batch_img_metas  = loaded_batch['img_metas'].data[0]
            with torch.no_grad():
                batch_features          = model.extract_feat(loaded_batch_img)

                # To get RoI features, we need to first generate region proposals. However, we may opt to use the ground-truth
                # bboxes when proposing regions so that a more accurate feature representation can be extracted for an object.
                # Here, region proposals (or ground truths) are extracted and passed to the Fast RCNN head (_get_RoI_features).
                if gt_proposals:
                    batch_proposals         = loaded_batch['gt_bboxes'].data[0]
                    num_proposals_per_img   = tuple(len(p) for p in batch_proposals)
                else:
                    batch_proposals         = model.rpn_head.simple_test_rpn(batch_features, loaded_batch_img_metas)
                    num_proposals_per_img   = tuple(len(p) for p in batch_proposals)
                
                fc_features, cls_scores = self._get_RoI_features(model, batch_features, batch_proposals)
                cls_softmaxes           = cls_scores.softmax(-1)

                # Split the feature tensors so that they are batched again.                
                fc_features             = fc_features.split(num_proposals_per_img, 0)
                cls_softmaxes           = cls_softmaxes.split(num_proposals_per_img, 0)

            # Filter out the top-k proposals per image
            for fc_feature_vectors, cls_softmax_vectors in zip(fc_features, cls_softmaxes):
                
                # Get the most-confident predictions for each proposal.
                max_score_per_proposal, pred_class_per_proposal = torch.max(cls_softmax_vectors, dim=1)
                
                # The last class predicted is assumed to be a background class. Since it doesn't reflect a class of foreground
                # objects, we preferrably would like to encode only information about foreground objects in the image. Hence, filter
                # out the background classes.
                predicted_fg_proposals      = pred_class_per_proposal != (cls_softmax_vectors.shape[1] - 1)
                fg_max_score_per_proposal   = max_score_per_proposal[predicted_fg_proposals]
                fg_fc_feature_vectors       = fc_feature_vectors[predicted_fg_proposals]

                # Sort the confidence of each prediction. Of these, select all of the fg proposals before
                # the bg proposals
                sorted_fg_proposal_scores   = torch.argsort(fg_max_score_per_proposal, descending=True)
                selected_fg_proposal_amt    = min(len(predicted_fg_proposals), self.max_prop_obj_det)
                kept_proposals              = fg_fc_feature_vectors[sorted_fg_proposal_scores[:selected_fg_proposal_amt]]

                # Store the proposals for this (and each) image in a list. A tensor will be formed once the minimum proposal count is
                # available so that a proper tensor can be formed (without padding)
                padded_kept_proposals                           = torch.zeros(self.max_prop_obj_det, kept_proposals.shape[1])
                padded_kept_proposals[:kept_proposals.shape[0]] = kept_proposals

                # Add these feature vectors to the extracted ones
                if extracted_feature_vectors is None:
                    extracted_feature_vectors = torch.zeros(len(dataset), self.max_prop_obj_det, kept_proposals.shape[1]).to(self.args["device"])
                
                extracted_feature_vectors[start_idx] = padded_kept_proposals
                start_idx += 1

        return extracted_feature_vectors


    def compute_sem_seg_similarity_kernel(self, n_d_features):

        print("Computing semantic segmentation similarity kernel")

        metric = self.args['metric'] if 'metric' in self.args else 'rbf'

        n, d = n_d_features.shape

        # Generate pairwise distances based on the embedding type. Here, we also attempt to normalize the distances to some degree
        # so that most similarity values in the kernel do not vanish (e^-20, for example, may as well be 0).
        pairwise_distances      = torch.cdist(n_d_features.to(self.device), n_d_features.to(self.device))
        rbf_sig                 = torch.max(pairwise_distances) / 4     # Div by 4 so that the max distance has z-score of 4.
        pw_square_distances     = torch.pow(pairwise_distances, 2)
        norm_pw_sq_distances    = pw_square_distances / (2. * rbf_sig * rbf_sig)
        full_sijs               = torch.exp(-norm_pw_sq_distances)

        return full_sijs


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
                # along the last dimension (the d-dimensional feature vectors). Then, compute the mean of
                # these similarities by only averaging those that do not correspond to zero-vector similarities.
                nbatch_n_k_k_kern                                   = torch.tensordot(batch_normalized_features, normalized_feature_tensor, dims=([-1],[-1])).permute(0,2,1,3)
                nbatch_n_image_kern                                 = torch.sum(nbatch_n_k_k_kern, dim=(2,3)) 
                nbatch_n_avg_denom                                  = torch.outer(batch_proposal_nonzero_vector_counts, image_proposal_nonzero_vector_counts)
                nbatch_n_image_kern                                 = nbatch_n_image_kern / nbatch_n_avg_denom
                image_image_similarity_kernel[start_idx:end_idx]    = nbatch_n_image_kern

                # Update the new start idx for the next iteration
                pbar.update(end_idx - start_idx)
                start_idx = end_idx

        # Lastly, since the values of the computed kernel could be between -1 and 1, do a simple fix to ensure those ranges are between 0 and 1.
        non_negative_image_image_similarity_kernel = (image_image_similarity_kernel + 1.) / 2.
        return non_negative_image_image_similarity_kernel


    @abstractmethod
    def select(self, budget):
        pass