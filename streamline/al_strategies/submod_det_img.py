from distil.utils.utils import LabeledToUnlabeledDataset

import submodlib
import torch

from .streamline_base_det import StreamlineBaseDetection

class SubmodularDetImage(StreamlineBaseDetection):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(SubmodularDetImage, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        self.num_tasks = len(labeled_dataset.task_idx_partitions)
        self.cfg                = args["cfg"]
        self.max_prop_obj_det   = 50


    def compute_obj_det_features(self, dataset, gt_proposals=False):

        dataloader  = build_dataloader(dataset, self.cfg["data"]["samples_per_gpu"], self.cfg["data"]["workers_per_gpu"], 
                                        num_gpus=1, dist=True, shuffle=False, seed=self.cfg["seed"], persistent_workers=False)
        model       = self.model.to(self.args["device"])

        # Extract backbone features
        extracted_feature_vectors   = None
        start_idx                   = 0
        for loaded_batch in dataloader:

            # Get the feature vectors at the feature-map level of the model pipeline. Get features throughout the forward process.
            loaded_batch_img        = loaded_batch['img'].data[0].to(self.args["device"]) # post-aug collated batch tensor is in ['img'][0]
            loaded_batch_img_metas  = loaded_batch['img_metas'].data[0]
            with torch.no_grad():
                batch_features          = torch.flatten(model.extract_feat(loaded_batch_img), start_dim=1)

                # Add these feature vectors to the extracted ones
                if extracted_feature_vectors is None:
                    extracted_feature_vectors = torch.zeros(len(dataset), batch_features.shape[1]).to(self.args["device"])
                
                end_idx                                         = min(start_idx + loaded_batch['img'].data[0].shape[0], len(dataset))
                extracted_feature_vectors[start_idx:end_idx]    = batch_features
                start_idx                                       = end_idx

        return extracted_feature_vectors


    def select(self, budget):
        
        self.model.eval()

        # Get the object detection similarity kernel, which will be used for selecting new instances
        obj_det_unlab_feat                                  = self.compute_obj_det_features(self.unlabeled_dataset, gt_proposals=False)
        obj_det_kern                                        = (submodlib.helper.create_kernel(X=obj_det_unlab_feat.cpu().numpy(), metric="cosine", method="sklearn") + 1.) / 2.

        # Use submodlib's fac loc function and maximize according to the budget
        obj = submodlib.FacilityLocationFunction(n=obj_det_kern.shape[0],
                                                    mode="dense",
                                                    separate_rep=False,
                                                    sijs=obj_det_kern)
        greedy_list = obj.maximize(budget=budget,
                                    optimizer="LazyGreedy",
                                    stopIfZeroGain=False,
                                    stopIfNegativeGain=False,
                                    verbose=False)
        greedy_indices = [x[0] for x in greedy_list]

        return greedy_indices