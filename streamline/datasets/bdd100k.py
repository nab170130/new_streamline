import json
from multiprocessing.sharedctypes import Value
import time
import numpy as np
import os
import tempfile
import torch

from torch.utils.data import Dataset

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.datasets.api_wrappers import COCO

import mmcv

class BDD100K(Dataset):

    def __init__(self, root_dir, train):
        self._initialize_dataset(root_dir, train)


    def _create_per_attr_idx(self, full_dataset):
        attribute_dict = {'weather': {'rainy': [], 'snowy':[], 'clear':[], 'overcast':[], 'undefined':[], 'partly cloudy':[], 'foggy':[]}, \
                    'scene': {'tunnel':[], 'residential':[], 'parking lot':[], 'undefined':[], 'city street':[], 'gas stations':[], 'highway':[]}, \
                    'timeofday': {'daytime':[], 'night':[], 'dawn/dusk':[], 'undefined':[]}}

        # Go through each instance in the passed dataset. Add the index corresponding to this 
        # instance to each of the above categories for each attribute.
        for i in range(len(full_dataset)):
            img_data = full_dataset.data_infos[i]
            img_attr = img_data["attributes"]
            attribute_dict['weather'][img_attr['weather']].append(i)
            attribute_dict['scene'][img_attr['scene']].append(i)
            attribute_dict['timeofday'][img_attr['timeofday']].append(i)

        # Return the attribute dictionary, which classifies each index to a particular attribute category.
        return attribute_dict


    def _initialize_dataset(self, root_dir, train):

        # BDD100K requires different treatment. We will use MMDetection to load the base dataset based on the configs given in the utils folder.
        bdd100k_config_path = "streamline/utils/mmdet_configs/faster_rcnn_r50_fpn_1x_bdd100k_cocofmt.py"
        bdd100k_config      = Config.fromfile(bdd100k_config_path)

        if train:
            base_dataset    = build_dataset(bdd100k_config.data.train['dataset'], None)
            annotation_path         = os.path.join(root_dir, "bdd_100k", "bdd100k", "labels", "det_20", "det_train_coco.json")
        else:
            base_dataset    = build_dataset(bdd100k_config.data.val, None)
            annotation_path         = os.path.join(root_dir, "bdd_100k", "bdd100k", "labels", "det_20", "det_val_coco.json")

        # Now that the base dataset is loaded, we shall create the task splits for each task. Here,
        # we formulate reasonable tasks that an autonomous vehicle may encounter:
        #
        #       1.  Rainy city streets, daytime
        #       2.  Clear city streets, nighttime
        #       3.  Foggy highways, daytime
        #       4.  Clear highways, nighttime
        #       5.  Clear residential, daytime
        #       6.  Clear residential, nighttime
        #
        # Before this is done, we first get an attribute dictionary that assigns each index to a particular attribute value
        attribute_tuple_to_idx_dict = self._create_per_attr_idx(base_dataset)

        task_attribute_tuples = ["daytime", "night"]

        # Form each task partition idx by finding the intersection across each attribute value. This will be the behind-the-scenes
        # mapping used for the BDD100k base dataset given here.
        base_mapping_partitions = []
        task_idx_partitions = []
        start_idx = 0
        for time_attr in task_attribute_tuples:    
            all_matching_time_idx       = set(attribute_tuple_to_idx_dict["timeofday"][time_attr])
            intersection_idx            = list(all_matching_time_idx) 
            end_idx                     = start_idx + len(intersection_idx)
            base_mapping_partitions.append(intersection_idx)
            task_idx_partition = list(range(start_idx, end_idx))
            task_idx_partitions.append(task_idx_partition)
            start_idx = end_idx

        # If this is the test set, we need to ensure balance across tasks.
        if not train:
            torch.manual_seed(40)
            np.random.seed(40)

            # Choose enough instances from each task. Redo the task_idx_partitions.
            min_task_test_size = min([len(x) for x in base_mapping_partitions])
            task_idx_partitions = []
            base_mapping = []
            start_idx = 0
            for base_mapping_partition in base_mapping_partitions:
                end_idx = start_idx + min_task_test_size
                selected_idx = np.random.choice(base_mapping_partition, size=min_task_test_size, replace=False).tolist()
                task_idx_partition = list(range(start_idx, end_idx))
                task_idx_partitions.append(task_idx_partition)
                start_idx = end_idx
                base_mapping.extend(selected_idx)

            # Change seed after sampling using current unix time. Modulo to be within 2**32 - 1.
            new_seed = time.time_ns() % 1000000
            torch.manual_seed(new_seed)
            np.random.seed(new_seed)

        else:

            base_mapping = []
            for base_mapping_partition in base_mapping_partitions:
                base_mapping.extend(base_mapping_partition)

        # Set fields for this object
        self.annotation_path        = annotation_path
        self.base_mapping   = base_mapping
        self.task_idx_partitions    = task_idx_partitions
        self.base_dataset           = base_dataset
        self.num_tasks              = len(task_idx_partitions)
        self.CLASSES                = self.base_dataset.CLASSES
        self._set_group_flag()


    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            base_idx        = self.base_mapping[i]
            self.flag[i]    = self.base_dataset.flag[base_idx]


    def get_task_number_and_index_in_task(self, index):

        # Determine to which partition this index belongs and the particular index within that task.
        working_idx = index
        for task_num, task_idx_partition in enumerate(self.task_idx_partitions):
            if working_idx - len(task_idx_partition) < 0:
                break
            working_idx = working_idx - len(task_idx_partition)
            
        return task_num, working_idx


    def __getitem__(self, index):
        
        # Get the index that corresponds to the base bdd100k dataset used.
        base_idx                = self.base_mapping[index]
        instance                = self.base_dataset[base_idx]

        return instance


    def __len__(self):
        return len(self.base_mapping)   


    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.
        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.
        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]


    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            
            # Modification from mmdet: Map idx to base dataset index!
            mapped_idx = self.base_mapping[idx]

            img_id = self.base_dataset.img_ids[mapped_idx]
            bboxes = results[mapped_idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results


    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):

            # Modification from mmdet: Map idx to base dataset index!
            mapped_idx = self.base_mapping[idx]

            img_id = self.base_dataset.img_ids[mapped_idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results


    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.
        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.
        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".
        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files


    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = os.path.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir


    def _create_result_gt_json(self):

        with open(self.annotation_path, "r") as all_annotation_file:
            all_anns = json.load(all_annotation_file)

        # Use the data_infos field to get only those image ids that are reference in the current dataset object
        referenced_image_ids = []
        for mapped_idx in self.base_mapping:
            referenced_id = self.base_dataset.data_infos[mapped_idx]["id"]
            referenced_image_ids.append(referenced_id)

        # Create a new json to store temporarily. It carries the same type and categories as the full annotation file
        result_gts_json = {}
        result_gts_json["type"]         = all_anns["type"]
        result_gts_json["categories"]   = all_anns["categories"]
        result_gts_json["images"]       = []
        result_gts_json["annotations"]  = []

        # Go through the full annotation file's images, keeping only those referenced image ids.
        for image_dict in all_anns["images"]:
            if image_dict["id"] in referenced_image_ids:
                result_gts_json["images"].append(image_dict)

        # Go through the full annotation file's annotations, keeping only those that reference one of our kept images
        for annotation_dict in all_anns["annotations"]:
            if annotation_dict["image_id"] in referenced_image_ids:
                result_gts_json["annotations"].append(annotation_dict)

        return result_gts_json


    def evaluate(self, results, metric='bbox', logger=None, jsonfile_prefix=None, classwise=False, proposal_nums=(100,300,1000), iou_thrs=None, metric_items=None):

        # Here, we add an evaluate function to match that of mmdetection. Since this object effectively takes a subset of the 
        # ACTUAL evaluation set, we need to make sure that the ground-truth annotations align with the results of inference
        # on this dataset. Here, we implement a slightly altered form of the CocoDataset definition from mmdet that maps an index
        # to the base dataset.

        # Unchanged from mmdetection
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        # Get a COCO-style annotation json that contains only those 
        # images/annotations corresponding to images used in this dataset object.
        # Save it to a temporary file so that it can be loaded/converted by COCO utility.
        result_gts_json = self._create_result_gt_json()
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, 'result_gts.json')
        with open(temp_file_path, "w") as temp_file:
            json.dump(result_gts_json, temp_file)
        coco_gt = COCO(temp_file_path)

        # Also unchanged
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.base_dataset.evaluate_det_segm(results, result_files, coco_gt,
                                                            metrics, logger, classwise,
                                                            proposal_nums, iou_thrs,
                                                            metric_items)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results