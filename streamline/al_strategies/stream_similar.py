from distil.active_learning_strategies import SMI
from distil.active_learning_strategies.strategy import Strategy

from distil.active_learning_strategies.strategy import Strategy
from distil.utils.utils import LabeledToUnlabeledDataset

from torch.utils.data import Subset

import submodlib
import torch

class StreamSimilar(Strategy):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):

        super(StreamSimilar, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)


    def select(self, budget):

        self.model.eval()
        
        # Utilize DISTIL's implementation of SIMILAR's SMI component. First, form the query set ALWAYS as the rare slice,
        # which by experiment design is the last slice.
        self.args['smi_function']   = "fl2mi"
        rare_start_idx              = len(self.labeled_dataset) - len(self.labeled_dataset.task_idx_partitions[-1])
        rare_end_idx                = len(self.labeled_dataset)
        rare_query_set              = Subset(self.labeled_dataset, list(range(rare_start_idx, rare_end_idx)))
        wrapped_similar             = SMI(self.labeled_dataset, self.unlabeled_dataset, rare_query_set, self.model, self.target_classes, self.args)

        # Select new unlabeled indices to add using SMI.
        selected_unlabeled_idx = wrapped_similar.select(budget)
        return selected_unlabeled_idx