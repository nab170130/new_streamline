from distil.active_learning_strategies import   BADGE, CoreSet, EntropySampling, FASS, GLISTER, LeastConfidenceSampling, \
                                                MarginSampling, PartitionStrategy, RandomSampling, GradMatchActive, SubmodularSampling

from .reservoir_wrapper import ReservoirWrapperStrategy
from .part_reservoir_wrapper import PartitionReservoirWrapperStrategy
from .simple_unlim_wrapper import SimpleUnlimitedMemoryWrapperStrategy

from .entropy_det import EntropyDetection
from .least_confidence_det import LeastConfidenceDetection
from .margin_det import MarginDetection
from .submod_det import SubmodularDet
from .submod_det_img import SubmodularDetImage

from .stream_similar import StreamSimilar
from .talisman import Talisman

from .lim_mem_streamline import LimitedMemoryStreamline
from .lim_mem_streamline_det import LimitedMemoryStreamlineDetection
from .unlim_mem_streamline import UnlimitedMemoryStreamline
from .unlim_mem_streamline_ablated_budget import UnlimitedMemoryStreamlineAblatedBudget
from .unlim_mem_streamline_ablated_scg import UnlimitedMemoryStreamlineAblatedSCG
from .unlim_mem_streamline_det import UnlimitedMemoryStreamlineDetection
from .unlim_mem_streamline_repl_scg import UnlimitedMemoryStreamlineReplacedSCG


class ALFactory:
    
    def __init__(self, train_dataset, unlabeled_dataset, model, num_classes, al_params):
        self.train_dataset = train_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.model = model
        self.num_classes = num_classes
        self.al_params = al_params
        
        
    def get_strategy(self, al_name):
        
        if al_name.startswith("entropy_det"):
            strategy_class = EntropyDetection
        elif al_name.startswith("margin_det"):
            strategy_class = MarginDetection
        elif al_name.startswith("least_confidence_det"):
            strategy_class = LeastConfidenceDetection
        elif al_name.startswith("submodular_det"):
            strategy_class = SubmodularDet
        elif al_name.startswith("submodular_det_img"):
            strategy_class = SubmodularDetImage
        elif al_name.startswith("lm_streamline_det"):
            strategy_class = LimitedMemoryStreamlineDetection
            smi_function = al_name.split("_")[3]
            obj_function = al_name.split("_")[4]
            self.al_params["smi_function"] = smi_function
            self.al_params["obj_function"] = obj_function
            if "oracle" in al_name:
                self.al_params["oracle_task_identity"] = True
            else:
                self.al_params["oracle_task_identity"] = False
        elif al_name.startswith("ulm_streamline_det"):
            strategy_class = UnlimitedMemoryStreamlineDetection
            smi_function = al_name.split("_")[3]
            obj_function = al_name.split("_")[4]
            self.al_params["smi_function"] = smi_function
            self.al_params["obj_function"] = obj_function
            if "oracle" in al_name:
                self.al_params["oracle_task_identity"] = True
            else:
                self.al_params["oracle_task_identity"] = False
        elif al_name.startswith("lm_streamline"):
            strategy_class = LimitedMemoryStreamline
            smi_function            = al_name.split("_")[2]
            identification_metric   = al_name.split("_")[3]
            obj_function            = al_name.split("_")[4]
            selection_metric        = al_name.split("_")[5]
            self.al_params["smi_function"]          = smi_function
            self.al_params["identification_metric"] = identification_metric
            self.al_params["obj_function"]          = obj_function
            self.al_params["selection_metric"]      = selection_metric
            if "oracle" in al_name:
                self.al_params["oracle_task_identity"] = True
            else:
                self.al_params["oracle_task_identity"] = False
        elif al_name.startswith("ulm_streamline"):
            strategy_class = UnlimitedMemoryStreamline
            smi_function            = al_name.split("_")[2]
            identification_metric   = al_name.split("_")[3]
            obj_function            = al_name.split("_")[4]
            selection_metric        = al_name.split("_")[5]
            self.al_params["smi_function"]          = smi_function
            self.al_params["identification_metric"] = identification_metric
            self.al_params["obj_function"]          = obj_function
            self.al_params["selection_metric"]      = selection_metric
            if "oracle" in al_name:
                self.al_params["oracle_task_identity"] = True
            else:
                self.al_params["oracle_task_identity"] = False
        elif al_name.startswith("ablated_budget_ulm_streamline"):
            strategy_class = UnlimitedMemoryStreamlineAblatedBudget
            smi_function            = al_name.split("_")[4]
            identification_metric   = al_name.split("_")[5]
            obj_function            = al_name.split("_")[6]
            selection_metric        = al_name.split("_")[7]
            self.al_params["smi_function"]          = smi_function
            self.al_params["identification_metric"] = identification_metric
            self.al_params["obj_function"]          = obj_function
            self.al_params["selection_metric"]      = selection_metric
            if "oracle" in al_name:
                self.al_params["oracle_task_identity"] = True
            else:
                self.al_params["oracle_task_identity"] = False
        elif al_name.startswith("ablated_scg_ulm_streamline"):
            strategy_class = UnlimitedMemoryStreamlineAblatedSCG
            smi_function            = al_name.split("_")[4]
            identification_metric   = al_name.split("_")[5]
            obj_function            = al_name.split("_")[6]
            selection_metric        = al_name.split("_")[7]
            self.al_params["smi_function"]          = smi_function
            self.al_params["identification_metric"] = identification_metric
            self.al_params["obj_function"]          = obj_function
            self.al_params["selection_metric"]      = selection_metric
            if "oracle" in al_name:
                self.al_params["oracle_task_identity"] = True
            else:
                self.al_params["oracle_task_identity"] = False
        elif al_name.startswith("repl_scg_ulm_streamline"):
            strategy_class = UnlimitedMemoryStreamlineReplacedSCG
            smi_function            = al_name.split("_")[4]
            identification_metric   = al_name.split("_")[5]
            obj_function            = al_name.split("_")[6]
            selection_metric        = al_name.split("_")[7]
            self.al_params["smi_function"]          = smi_function
            self.al_params["identification_metric"] = identification_metric
            self.al_params["obj_function"]          = obj_function
            self.al_params["selection_metric"]      = selection_metric
            if "oracle" in al_name:
                self.al_params["oracle_task_identity"] = True
            else:
                self.al_params["oracle_task_identity"] = False
        elif al_name.startswith("badge"):

            # BADGE relies on last-layer gradients, which can be large for some dataset/model combos.
            # We currently assume that the training dataset is simply the transform subset wrapped around a base one.
            base_dataset_type = type(self.train_dataset.base_dataset)
            if base_dataset_type.__name__ == "CIFAR100":
                self.al_params["num_partitions"] = 3
                self.al_params["wrapped_strategy_class"] = BADGE
                strategy_class = PartitionStrategy
            else:
                strategy_class = BADGE
        elif al_name.startswith("coreset"):
            strategy_class = CoreSet
        elif al_name.startswith("entropy"):
            strategy_class = EntropySampling
        elif al_name.startswith("fass"):
            strategy_class = FASS
        elif al_name.startswith("glister"):

            # GLISTER also relies on last-layer gradients, which can be large for some dataset/model combos.
            # We currently assume that the training dataset is simply the transform subset wrapped around a base one.
            base_dataset_type = type(self.train_dataset.base_dataset)
            if base_dataset_type.__name__ == "CIFAR100":
                self.al_params["num_partitions"] = 4
                self.al_params["wrapped_strategy_class"] = GLISTER
                strategy_class = PartitionStrategy
            else:
                strategy_class = GLISTER
        elif al_name.startswith("gradmatch"):
            strategy_class = GradMatchActive
        elif al_name.startswith("least_confidence"):
            strategy_class = LeastConfidenceSampling
        elif al_name.startswith("margin"):
            strategy_class = MarginSampling
        elif al_name.startswith("random"):
            strategy_class = RandomSampling
        elif al_name.startswith("submodular"):
            strategy_class = SubmodularSampling
        elif al_name.startswith("stream_similar"):
            strategy_class = StreamSimilar
        elif al_name.startswith("talisman"):
            strategy_class = Talisman
        else:
            raise ValueError(F"AL strategy {al_name} not supported")
            
        if al_name.endswith("_part_reservoir"):
            self.al_params["wrapped_al_strategy"] = strategy_class
            strategy = PartitionReservoirWrapperStrategy(self.train_dataset, self.unlabeled_dataset, self.model, self.num_classes, self.al_params)
        elif al_name.endswith("_reservoir"):
            self.al_params["wrapped_al_strategy"] = strategy_class
            strategy = ReservoirWrapperStrategy(self.train_dataset, self.unlabeled_dataset, self.model, self.num_classes, self.al_params)
        elif "_streamline_" in al_name:
            strategy = strategy_class(self.train_dataset, self.unlabeled_dataset, self.model, self.num_classes, self.al_params)
        else:
            base_strategy = strategy_class(self.train_dataset, self.unlabeled_dataset, self.model, self.num_classes, self.al_params)
            strategy = SimpleUnlimitedMemoryWrapperStrategy(base_strategy)

        return strategy