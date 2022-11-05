from distil.active_learning_strategies.strategy import Strategy
from mmdet.core import bbox2roi
from mmseg.datasets import build_dataloader

import torch

class LeastConfidenceDetection(Strategy):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(LeastConfidenceDetection, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        self.num_tasks          = len(labeled_dataset.task_idx_partitions)
        self.cfg                = args["cfg"]
        self.max_prop_obj_det   = 100

    
    def _get_RoI_features(self, model, features, proposals):

        # Convert proposals to RoIs and compute flattened bounding box features
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


    def compute_average_confidence_scores(self, dataset):

        dataloader  = build_dataloader(dataset, self.cfg["data"]["samples_per_gpu"], self.cfg["data"]["workers_per_gpu"], 
                                        num_gpus=1, dist=True, shuffle=False, seed=self.cfg["seed"], persistent_workers=False)
        model       = self.model.to(self.args["device"])

        # Start computing the average of the max-entropic class predictions for each proposal in all images.
        average_prediction_confidence_scores    = torch.zeros(len(dataset)).to(self.args["device"])
        start_idx                               = 0
        for loaded_batch in dataloader:

            # Get the feature vectors at the feature-map level of the model pipeline. Get features throughout the forward process.
            loaded_batch_img        = loaded_batch['img'].data[0].to(self.args["device"]) # post-aug collated batch tensor is in ['img'][0]
            loaded_batch_img_metas  = loaded_batch['img_metas'].data[0]
            with torch.no_grad():
                batch_features          = model.extract_feat(loaded_batch_img)            
                batch_rpn_proposals     = model.rpn_head.simple_test_rpn(batch_features, loaded_batch_img_metas)
                fc_features, cls_scores = self._get_RoI_features(model, batch_features, batch_rpn_proposals)
                cls_softmaxes           = cls_scores.softmax(-1)

                # Split the feature tensors so that they are batched again.
                num_proposals_per_img   = tuple(len(p) for p in batch_rpn_proposals)
                fc_features             = torch.stack(fc_features.split(num_proposals_per_img, 0))
                cls_softmaxes           = torch.stack(cls_softmaxes.split(num_proposals_per_img, 0))

            # Compute the maximum entropy score for a class by treating each class probability as a 
            # Bernoulli variable. Recall that the Faster-RCNN architecture uses Fast-RCNN, which predicts
            # a bounding box for each class per proposal. Hence, the probabilities correspond to whether or 
            # not the box actually should be drawn for that proposal.
            for cls_softmax_vectors in cls_softmaxes:
                
                predicted_class_confidence_per_proposal, _      = torch.max(cls_softmax_vectors, dim=1)
                average_prediction_confidence                   = torch.mean(predicted_class_confidence_per_proposal)
                average_prediction_confidence_scores[start_idx] = average_prediction_confidence
                start_idx += 1

        return average_prediction_confidence_scores


    def select(self, budget):
        
        # Get the average confidence score of the most likely class across all RoIs for each 
        # image. Choose those images that have the smallest such prediction confidence.
        acquisition_scores  = self.compute_average_confidence_scores(self.unlabeled_dataset)
        score_ranking_idx   = torch.argsort(acquisition_scores, descending=False)
        selected_idx        = score_ranking_idx[:budget]

        return selected_idx 