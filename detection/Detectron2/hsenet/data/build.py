# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
from tabulate import tabulate
from termcolor import colored

from detectron2.config import configurable
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import log_first_n

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler

## Use original detectron2 functions
from detectron2.data.build import (
    build_batch_data_loader,
    print_instances_class_histogram,
    trivial_batch_collator,
    worker_init_reset_seed,
    #filter_images_with_only_crowd_annotations,
    load_proposals_into_dataset,
    #filter_images_with_few_keypoints,
)

## User functions
from .detection_utils import check_metadata_consistency


"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    #"build_batch_data_loader",
    "build_detection_train_loader",
    "build_detection_test_loader",
    #"get_detection_dataset_dicts",
    #"load_proposals_into_dataset",
    #"print_instances_class_histogram",
]
def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with none annotations
    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        if len(anns)>0: return True
        #for ann in anns:
            #if ann.get("iscrowd", 0) == 0:
                #return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts

def filter_images_with_few_keypoints(dataset_dicts, min_keypoints_per_image):
    """
    Filter out images with too few number of keypoints.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format as dataset_dicts, but filtered.
    """
    num_before = len(dataset_dicts)

    def visible_keypoints_in_image(dic):
        # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility
        annotations = dic["annotations"]
        return sum(
            (np.array(ann["keypoints"][2::3]) > 0).sum()
            for ann in annotations
            if "keypoints" in ann
        )
    ### If dataset_dict is from  coco instance dataset(i.e. x["dataset_source"==3]) preserve
    ### If dataset_dict is from  mot dataset(i.e. x["dataset_source"==7]) preserve - don't have key points
    # append keypoints if annotation unexists
    for annotations in dataset_dicts:
        for ann in annotations["annotations"]:
            if not "keypoints" in ann.keys():
                ann["keypoints"] = np.ones(51)
    dataset_dicts = [
        x for x in dataset_dicts if visible_keypoints_in_image(x) >= min_keypoints_per_image or x["dataset_source"]==3 or x["dataset_source"]==7
    ]
    ###
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    print('The name is ', __name__)
    logger.info(
        "Removed {} images with fewer than {} keypoints.".format(
            num_before - num_after, min_keypoints_per_image
        )
    )
    return dataset_dicts
#########################################   <ADD>
def filter_images_without_next_tracking_boxes(dataset_dicts):
    num_before = len(dataset_dicts)
    def has_next_tracking_boxes(t1, t2):
        if t1['sequence_name'] != t2['sequence_name']:
            return False
        t1_annos = t1['annotations']
        t2_annos = t2['annotations']

        matching = []
        for i, t1_anno in enumerate(t1_annos):
            for j, t2_anno in enumerate(t2_annos):
                if t1_anno['track_id'] == t2_anno['track_id']:
                    matching.append((i, j))
                    break

        # remove non matching boxes
        # cur_frame['annotations'] = [cur_annos[pair[0]] for pair in matching]
        # next_frame['annotations'] = [next_annos[pair[1]] for pair in matching]

        if len(matching) == 0:
            return False

        return True
    # MOT dataset만 검사
    dataset_dicts = [
        dataset_dicts[i] for i in range(len(dataset_dicts)-1) if dataset_dicts[i]['dataset_source'] != 7 or has_next_tracking_boxes(dataset_dicts[i], dataset_dicts[i+1])
    ]
    
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images without next tracking boxes. {} images left.".format(num_before - num_after, num_after)
    )
    return dataset_dicts

def reassign_tracking_id(dataset_dicts, config_num_ids):
    # find all tracking ids per sequence
    seq_name_dict = []  # name List
    tid_set = dict()
    for img in dataset_dicts:
        if img['dataset_source'] != 7:
            continue
        seq_name = img['sequence_name']
        if seq_name not in tid_set:
            tid_set[seq_name] = set()
            seq_name_dict.append(seq_name)
        tids = [anno['track_id'] for anno in img['annotations']]
        tid_set[seq_name].update(tids)
    nds = [len(x) for x in tid_set.values()]            # 각 frame별 track_id 개수
    cds = [sum(nds[:i]) for i in range(len(nds))]       # offset. ex) [54, 83, 125, 25, 54, 69, 107] -> [0, 54, 137, 262, 287, 341, 410]

    # generate mapping dict from original tracking id to new id
    tid_map, offset = dict(), 0
    for i, seq_name in enumerate(seq_name_dict):
        tid_map[seq_name] = dict()
        offset = cds[i]
        for tid, new_tid in zip(tid_set[seq_name], range(0, nds[i] + 1)):
            tid_map[seq_name][tid] = new_tid + offset

    # remap tracking ids into dataset_dicts
    for img in dataset_dicts:
        if img['dataset_source'] != 7:
            continue
        seq_name = img['sequence_name']
        seq_tid_map = tid_map[seq_name]
        for anno in img['annotations']:
            tid = anno['track_id']
            new_tid = seq_tid_map[tid]
            anno['track_id'] = new_tid

    num_ids = sum(nds)
    logger = logging.getLogger(__name__)
    if config_num_ids is not None:
        logger.info("Number of tracking IDs in config: {}".format(config_num_ids))
    logger.info("Number of tracking IDs in dataset: {}".format(num_ids))

    return dataset_dicts, num_ids

def print_instances_track_id_histogram(dataset_dicts):
    seq_info = dict()
    for annotations_per_frame in dataset_dicts:
        if 'sequence_name' not in dataset_dicts[0]:
            continue
        seq_name = annotations_per_frame['sequence_name']
        if seq_name not in seq_info:
            seq_info[seq_name] = {"num_frames": 0, "tid_set": set(), "max_obj": 0, "total_obj": 0}
        tids = [annotation['track_id'] for annotation in annotations_per_frame['annotations']]
        seq_info[seq_name]["tid_set"].update(tids)
        seq_info[seq_name]["max_obj"] = max(len(annotations_per_frame['annotations']), seq_info[seq_name]["max_obj"])
        seq_info[seq_name]["total_obj"] += len(annotations_per_frame['annotations'])
        seq_info[seq_name]["num_frames"] += 1
    
    if not len(seq_info):
        return

    seq_names = seq_info.keys()
    num_frames = [seq_info[k]["num_frames"] for k in seq_info.keys()]
    nds = [len(seq_info[k]["tid_set"]) for k in seq_info.keys()]
    max_objs = [seq_info[k]["max_obj"] for k in seq_info.keys()]
    avg_objs = [seq_info[k]["total_obj"] / seq_info[k]["num_frames"] for k in seq_info.keys()]

    N_COLS = 5
    data = list(
        itertools.chain(
            *[[seq_name, int(num_frame), int(max_obj), f"{avg_obj:.1f}", int(num_ids)] for
              seq_name, num_frame, max_obj, avg_obj, num_ids in
              zip(seq_names, num_frames, max_objs, avg_objs, nds)]
        ))
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["sequence name", "#frame", "#max obj", "#avg obj", "#id"],
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    txt = "Distribution of instances among all sequences:\n" + colored(table, "cyan")
    print(txt)
    log_first_n(logging.INFO, txt, key="message")

#########################################   </ADD>
def get_detection_dataset_dicts(
    dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None,
    reassign_id=False, filter_pairless=True, num_ids=0
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]
    total_ids = num_ids
    dataset_dicts_list = dataset_dicts
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)

    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)
        pass
   
    if filter_pairless and has_instances:
        dataset_dicts = filter_images_without_next_tracking_boxes(dataset_dicts)
    
    if reassign_id and has_instances:
        dataset_dicts, total_ids = reassign_tracking_id(dataset_dicts, num_ids)

    if has_instances:
        try:
            check_metadata_consistency("thing_classes", dataset_names)
            for ii in range(len(dataset_names)):
                class_names = MetadataCatalog.get(dataset_names[ii]).thing_classes
                print_instances_class_histogram(dataset_dicts_list[ii], class_names)
                print_instances_track_id_histogram(dataset_dicts_list[ii])   
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts, total_ids


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset, total_ids = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            filter_pairless=cfg.DATALOADER.FILTER_PAIRLESS_ANNOTATIONS if cfg.MODEL.TRACKING_ON else False,
            reassign_id = True if cfg.MODEL.TRACKING_ON and hasattr(cfg.DATALOADER, "NUM_IDS") else False,
            num_ids = cfg.DATALOADER.NUM_IDS if cfg.MODEL.TRACKING_ON and hasattr(cfg.DATALOADER, "NUM_IDS") else 0
        )   # if not tracking, total_ids=0
    if total_ids != 0:
        cfg.DATALOADER.defrost()
        cfg.DATALOADER.NUM_IDS = total_ids
        cfg.DATALOADER.freeze()
    
    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


# TODO can allow dataset as an iterable or IterableDataset to make this function more general
@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    dataset, _ = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {"dataset": dataset, "mapper": mapper, "num_workers": cfg.DATALOADER.NUM_WORKERS}


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *, mapper, num_workers=0):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

def convert_mot2coco(dataset_name, dataset_dicts):
    meta = MetadataCatalog.get(dataset_name)
    thing_classes = meta.thing_classes
    has_instances = 'annotations' in dataset_dicts[0]

    images = []
    annotations = []
    for i, data in enumerate(dataset_dicts):
        images.append({
            'file_name': data['file_name'],
            'id': data['image_id'],
            'height': int(data['height']),
            'width': int(data['width']),
        })
        if has_instances:
            for anno in data['annotations']:
                xywh = [int(coord) for coord in anno['bbox']]
                annotations.append({
                    'id': anno['id'],
                    'category_id': anno['category_id'],
                    'image_id': i,
                    'bbox': xywh,
                    'area': xywh[2] * xywh[3],
                    'iscrowd': 0,
                })

    categories = []
    for i, thing_class in enumerate(thing_classes):
        categories.append({
            "supercategory": thing_class,
            "id": i,
            "name": thing_classes,
        })

    results = {'images': images, 'categories': categories}
    if has_instances:
        results.update({'annotations': annotations})
    return results