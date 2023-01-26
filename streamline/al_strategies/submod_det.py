from distil.utils.utils import LabeledToUnlabeledDataset

import submodlib
import torch

from .streamline_base_det import StreamlineBaseDetection

class SubmodularDet(StreamlineBaseDetection):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(SubmodularDet, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        self.num_tasks = len(labeled_dataset.task_idx_partitions)
        self.cfg                = args["cfg"]
        self.max_prop_obj_det   = 50


    def select(self, budget):
        
        self.model.eval()

        # Get the object detection similarity kernel, which will be used for selecting new instances
        obj_det_unlab_feat                                  = self.compute_obj_det_features(self.unlabeled_dataset, gt_proposals=False)
        obj_det_kern                                        = self.compute_obj_det_similarity_kernel(obj_det_unlab_feat).cpu().numpy()

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