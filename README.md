# Streamline

Contains code used in experiment results of *Streamline: Streaming Active Learning for Realistic Multi-Distributional Settings*. Here, a brief description is provided for reproduction of the presented results.

## Installation

The Github repository can be cloned to a target machine. Dependencies are managed via Conda -- within [conda_env_packages.txt](conda_env_packages.txt) is the full set of dependencies used to produce the results. Additionally, [DISTIL](https://github.com/decile-team/distil) is used for base active learning methods.

## Datasets

This code will automatically download the datasets used in the image classification experiments; however, the object detection datasets will require manual setup before they can be used. Notably, the interfaces developed here use the COCO annotation format for each dataset, and extra utilities are provided here to help generate these COCO annotations.

### BDD-100K

BDD-100K can be downloaded from the following [link](https://www.bdd100k.com/). Additionally, code utilities for generating the requisite COCO annotations are provided at that link as well. The following directory structure is expected in the root data folder:

```
bdd_100k/
    bdd100k/
        images/
            100k/
                test/
                    ...
                train/
                    ...
                val/
                    ...
        labels/
            det_20/
                det_train_coco.json
                det_val_coco.json
```

### Cityscapes

Cityscapes can be downloaded from the following [link](https://www.cityscapes-dataset.com/). Additionally, [MMDetection](https://github.com/open-mmlab/mmdetection/tree/master/configs/cityscapes) provide a conversion script for generating the base COCO labels for Cityscapes.

To produce rare-slice data, we utilize the techniques from *Physics-Based Rendering for Improving Robustness to Rain* by Halder *et al*. The devkit and requisite data can be obtained [here](https://team.inria.fr/rits/computer-vision/weather-augment/). After the modified Cityscapes images are generated, the final COCO label files (base_rain_coco_train.json, base_rain_coco_val.json) can be generated using the [raincoco.py](dataset_utils/raincoco.py) file. Ultimately, the dataset interfaces used here expect the following directory structure in the root data folder:

```
cityscapes/
    gtFine/
    leftImg8bit/
        test/
        train/
            200mm/
                aachen/
                    ...
                bochum/
                    ...
                ...
            base/
                aachen/
                    ...
                bochum/
                    ...
                ...
        val/
            200mm/
                frankfurt/
                    ...
                lindau/
                    ...
                munster/
                    ...
            base/
                frankfurt/
                    ...
                lindau/
                    ...
                munster/
                    ...
    base_rain_coco_train.json
    base_rain_coco_val.json
    instancesonly_filtered_gtFine_test.json
    instancesonly_filtered_gtFine_train.json
    instancesonly_filtered_gtFine_val.json
```

### KITTI

KITTI can be downloaded from the following [link](https://www.cvlibs.net/datasets/kitti/). A COCO label conversion script is given in [tococo_kitti.py](dataset_utils/tococo_kitti.py). 

To produce rare-slice data, we again utilize the techniques from *Physics-Based Rendering for Improving Robustness to Rain* by Halder *et al*. The devkit and requisite data can be obtained [here](https://team.inria.fr/rits/computer-vision/weather-augment/). After the modified KITTI images are generated, the final COCO label file (coco_annotations_foggy.json) can be generated using the [tococo_kitti.py](dataset_utils/tococo_kitti.py) file. Ultimately, the dataset interfaces used here expect the following directory structure in the root data folder:

```
kitti/
    data_object/
        training/
            image_2/
                base/
                    ...
                fog/
                    30m/
                        ...
            label_2/
                ...
            coco_annotations_foggy.json
```

## Results Folder

Results are stored in a folder of the following structure:

```
results/
    large_objects/
        lr_states/
        opt_states/
        splits/
        weights/
    streamline_db.db
```

## Pretrained Weights

The codebase expects [these PyTorch pretrained DenseNet weights](https://download.pytorch.org/models/densenet161-8d451a50.pth) to be within the `results/large_objects/weights` folder when using DenseNet-161. Additionally, all object detection experiments use [these MMSegmentation pretrained PSPNet weights](https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth), which are stored in the [streamline/utils/mmseg_configs/pspnet](streamline/utils/mmseg_configs/pspnet) folder.

## Config Files

The codebase used here operates off of configuration files. Before experiments can be executed, the `data_root`, `ann_file`, and `img_prefix` fields within [streamline/utils/mmdet_configs/faster_*](streamline/utils/mmdet_configs) will need to be updated to reflect the installation of the object detection datasets mentioned previously.

Experiments, metric computation, and plots are generated via configuration files. Examples are given in the [configs](configs) folder. The fields for each type are given here.

### Experiment Configs

The fields of the experiment config files are as follows:

| Field | Desc |
| ----- | ---- |
| gpus | A list of Pytorch GPU names that will be used in round-robin fashion to execute the following experiments |
| dataset_directory | The root directory where each dataset is stored |
| db_loc | The location of the results database file. Sqlite3 is used to store experiment progress, results, and other information used by the codebase |
| base_exp_directory | The location where large, intermediate objects will be stored. This is the `large_objects` folder mentioned above.|
| distil_directory | The path to the DISTIL repository mentioned above. |
| experiments | A list of lists that detail which experiments should be run under this configuration. Each inner list details the following: `[train_dataset, model_arch, 0, arrival_pattern, run_number, training_loop, active_learning_method, al_budget, initial_slice_size, unlabeled_stream_size, batch_size, num_al_rounds, delete_large_objects_after_run]`. Possible options for training loop, model architecture, and active learning method can be found in their respective code locations. Lastly, the `delete_large_objects_after_run` field will automatically evaluate every metric once the run is completed and will subsequently delete the large object files generated intermittently, which can be useful on space-constrained systems.

### Metric Configs

The fields of the metric config files are as follows:

| Field | Desc |
| ----- | ---- |
| gpus | A list of Pytorch GPU names that will be used in round-robin fashion to execute the following experiments |
| batch_size | The batch size to use when computing metrics |
| dataset_directory | The root directory where each dataset is stored |
| db_loc | The location of the results database file. Sqlite3 is used to store experiment progress, results, and other information used by the codebase |
| base_exp_directory | The location where large, intermediate objects will be stored. This is the `large_objects` folder mentioned above.|
| distil_directory | The path to the DISTIL repository mentioned above. |
| metrics | A list of lists that detail which metrics to compute under this metric computation. Each inner list is marked by `[train_dataset, metric_name, evaluation_dataset]`. Possible values for each are found in their respective code locations.

### Plot Configs

The fields of the plot config files depend on the *type* of plot being generated; however, these config files follow closely off of the experiment config files. The only additional fields added per-line are fields related to the line styling and color of various methods.

## Running Experiments

After setup, running experiments is made simple by executing the following:

```
python driver.py --create_db configs/experiments/*
```

where the `--create_db` flag can be omitted if a results database used by the passed config already exists. Notably, execution can be stopped intermittently, and the code can pick up from the last checkpointed spot in the experiment using the intermediate large objects. If the `delete_large_objects_after_run` field is set, then `driver.py` also computes every metric before finishing a run. If this field is not set, then (specific) metrics can be computed using the following:

```
python metrics.py configs/metrics/*
```

Finally, plots can be generated using the following:

```
python plotter.py configs/plots/*
```

## Note about Experiments

In general, we provide the full list of experiments ran in the config files of this repository -- only the directories within them need to be changed to match the target system that will execute the experiments.