from PIL import Image

import json
import os
import tqdm


def prepare_annotation_map(base_directory):
    
    annotation_directory = os.path.join(base_directory, "label_2")

    # List every text file in the annotation directory
    image_num_to_annotation_map = {}
    for filename in tqdm.tqdm(os.listdir(annotation_directory), "Loading KITTI Annotations"):
        abs_path = os.path.join(annotation_directory, filename)
        with open(abs_path, "r") as annotation_file:

            # Each annotation file contains rows for each bbox annotation
            annotation_strings   = annotation_file.readlines()
            all_annotation_tuples = []
            for annotation_string in annotation_strings:
                annotation_fields   = annotation_string.split(" ")
                
                class_name          = annotation_fields[0]
                truncated           = float(annotation_fields[1])
                occluded            = int(annotation_fields[2])
                alpha               = float(annotation_fields[3])
                bbox_x1             = float(annotation_fields[4])
                bbox_y1             = float(annotation_fields[5])
                bbox_x2             = float(annotation_fields[6])
                bbox_y2             = float(annotation_fields[7])
                height              = float(annotation_fields[8])
                width               = float(annotation_fields[9])
                length              = float(annotation_fields[10])
                location_x          = float(annotation_fields[11])
                location_y          = float(annotation_fields[12])
                location_z          = float(annotation_fields[13])
                rotation_y          = float(annotation_fields[14])

                if class_name != "DontCare":
                    coco_relevant_annotation_tuple = (class_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    all_annotation_tuples.append(coco_relevant_annotation_tuple)
        
        image_num = filename.split(".")[0]
        image_num_to_annotation_map[image_num] = all_annotation_tuples
    
    return image_num_to_annotation_map


def prepare_categories_field(image_num_to_annotation_map):
    
    print("Preparing COCO categories field")

    # Get all unique classes in the annotation map
    all_classes = set()
    for filename in image_num_to_annotation_map:
        file_annotations = image_num_to_annotation_map[filename]
        for class_name, _, _, _, _ in file_annotations:
            all_classes.add(class_name)
    
    # Prepare the COCO categories field, which is a list of dictionaries for each class. Additionally, prepare an easy
    # mapping from category name to category id for annotation generation.
    categories_field = []
    category_name_to_id_map = {}
    for i, class_name in enumerate(all_classes):
        category_dict = {"id": i, "name": class_name}
        categories_field.append(category_dict)
        category_name_to_id_map[class_name] = i

    return categories_field, category_name_to_id_map


def prepare_images_field(base_directory, modified_subfolder_path, image_num_to_annotation_map):

    # Prepare the images field, which is a list of dictionaries for each image.
    # Additionally, prepare an easy mapping from image number to image id for annotation generation.
    images_field = []
    filename_to_image_id_map = {}
    id = 0

    for image_num in tqdm.tqdm(image_num_to_annotation_map, "Preparing images field: Normal Images", len(image_num_to_annotation_map.keys())):
        file_name       = os.path.join("image_2", "base", F"{image_num}.png")
        abs_path        = os.path.join(base_directory, file_name)
        image           = Image.open(abs_path, "r")
        width, height   = image.size
        image_dict      = {"id": id, "width": width, "height": height, "file_name": file_name}
        images_field.append(image_dict)
        
        filename_to_image_id_map[file_name] = id
        id = id + 1

    for image_num in tqdm.tqdm(image_num_to_annotation_map, "Preparing images field: Mod. Images", len(image_num_to_annotation_map.keys())):
        file_name       = os.path.join("image_2", modified_subfolder_path, F"{image_num}.png")
        abs_path        = os.path.join(base_directory, file_name)
        image           = Image.open(abs_path, "r")
        width, height   = image.size
        image_dict      = {"id": id, "width": width, "height": height, "file_name": file_name}
        images_field.append(image_dict)
        
        filename_to_image_id_map[file_name] = id
        id = id + 1

    return images_field, filename_to_image_id_map


def prepare_annotations_field(base_directory, image_num_to_annotation_map, modified_subfolder_path, category_name_to_id_map, filename_to_image_id_map):

    # Prepare the annotations field, which is a list of dictionaries for each annotation.
    annotations_field = []
    anno_id = 0

    for image_num in tqdm.tqdm(image_num_to_annotation_map, "Preparing annotations field: Normal images", len(image_num_to_annotation_map.keys())):
        file_name = os.path.join("image_2", "base", F"{image_num}.png")
        for class_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2 in image_num_to_annotation_map[image_num]:

            width   = bbox_x2 - bbox_x1
            height  = bbox_y2 - bbox_y1

            annotation  = {}
            id          = anno_id
            image_id    = filename_to_image_id_map[file_name]
            category_id = category_name_to_id_map[class_name]
            area        = width * height
            bbox        = [bbox_x1, bbox_y1, width, height]
            iscrowd     = 0
            annotation  = {"id": id, "image_id": image_id, "category_id": category_id, "area": area, "bbox": bbox, "iscrowd": iscrowd}
            
            annotations_field.append(annotation)
            anno_id     = anno_id + 1

    for image_num in tqdm.tqdm(image_num_to_annotation_map, "Preparing annotations field: Mod. images", len(image_num_to_annotation_map.keys())):
        file_name = os.path.join("image_2", modified_subfolder_path, F"{image_num}.png")
        
        abs_path                    = os.path.join(base_directory, os.path.join("image_2", "base", F"{image_num}.png"))
        image                       = Image.open(abs_path, "r")
        image_width, image_height   = image.size

        for class_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2 in image_num_to_annotation_map[image_num]:

            # The bounding box for the modified images requires a slight modification to be consistent with the modified KITTI images.
            # The original conversion does a center crop of size 1216 x 352, so the bbox of these images needs to be relativized.
            center_crop_width   = 1216
            center_crop_height  = 352
            x_shift = center_crop_width // 2 - image_width // 2
            y_shift = center_crop_height // 2 - image_height // 2

            # Apply the shift, clipping so that the bbox does not go past the bounds of the image.
            bbox_x1 = min(max(bbox_x1 + x_shift, 0), center_crop_width)
            bbox_x2 = min(max(bbox_x2 + x_shift, 0), center_crop_width)
            bbox_y1 = min(max(bbox_y1 + y_shift, 0), center_crop_height)
            bbox_y2 = min(max(bbox_y2 + y_shift, 0), center_crop_height)

            width   = bbox_x2 - bbox_x1
            height  = bbox_y2 - bbox_y1

            annotation  = {}
            id          = anno_id
            image_id    = filename_to_image_id_map[file_name]
            category_id = category_name_to_id_map[class_name]
            area        = width * height
            bbox        = [bbox_x1, bbox_y1, width, height]
            iscrowd     = 0
            annotation  = {"id": id, "image_id": image_id, "category_id": category_id, "area": area, "bbox": bbox, "iscrowd": iscrowd}
            
            annotations_field.append(annotation)
            anno_id     = anno_id + 1
    
    return annotations_field


def prepare_rainy_coco_file(directory, image_num_to_annotation_map, categories_field, category_name_to_id_map):

    modified_subfolder = "rain/200mm/rainy_image"

    images_field, filename_to_image_id_map      = prepare_images_field(directory, modified_subfolder, image_num_to_annotation_map)
    annotations_field                           = prepare_annotations_field(directory, image_num_to_annotation_map, modified_subfolder, category_name_to_id_map, filename_to_image_id_map)

    coco_json = {"images": images_field, "annotations": annotations_field, "categories": categories_field}

    save_coco_dir = os.path.join(directory, F"coco_annotations_rain.json")
    with open(save_coco_dir, "w") as write_coco_file:
        json.dump(coco_json, write_coco_file)


def prepare_foggy_coco_file(directory, image_num_to_annotation_map, categories_field, category_name_to_id_map):

    modified_subfolder = "fog/30m"
    images_field, filename_to_image_id_map      = prepare_images_field(directory, modified_subfolder, image_num_to_annotation_map)
    annotations_field                           = prepare_annotations_field(directory, image_num_to_annotation_map, modified_subfolder, category_name_to_id_map, filename_to_image_id_map)

    coco_json = {"images": images_field, "annotations": annotations_field, "categories": categories_field}

    save_coco_dir = os.path.join(directory, F"coco_annotations_foggy.json")
    with open(save_coco_dir, "w") as write_coco_file:
        json.dump(coco_json, write_coco_file)


directory = "data_object/training"
image_num_to_annotation_map                 = prepare_annotation_map(directory)
categories_field, category_name_to_id_map   = prepare_categories_field(image_num_to_annotation_map)

prepare_rainy_coco_file(directory, image_num_to_annotation_map, categories_field, category_name_to_id_map)
prepare_foggy_coco_file(directory, image_num_to_annotation_map, categories_field, category_name_to_id_map)