#!/usr/bin/env python

import argparse
import random
import numpy as np

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from models.fewshot_anom import FewShotSeg
from dataloading.datasets import TestDataset
from dataloading.dataset_specifics import *
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--pretrained_root', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--all_slices', default=False, type=bool)
    parser.add_argument('--EP1', default=False, type=bool)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--workers', default=0, type=int)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Deterministic setting for reproducability.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Set up logging.
    logger = set_logger(args.save_root, 'train.log')
    logger.info(args)

    # Setup the path to save.
    args.save = os.path.join(args.save_root)

    # Init model and load state_dict.
    model = FewShotSeg(use_coco_init=False)
    model = nn.DataParallel(model.cuda())
    model.load_state_dict(torch.load(args.pretrained_root, map_location="cpu"))

    # Data loader.
    test_dataset = TestDataset(args)
    query_loader = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True)

    # Inference.
    logger.info('  Start inference ... Note: EP1 is ' + str(args.EP1))
    logger.info('  Support: ' + str(test_dataset.support_dir[len(args.data_root):]))
    logger.info('  Query: ' +
                str([elem[len(args.data_root):] for elem in test_dataset.image_dirs]))

    # Get unique labels (classes).
    labels = get_label_names(args.dataset)

    # Loop over classes.
    class_dice = {}
    class_iou = {}
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name is 'BG':
            continue

        logger.info('  *------------------Class: {}--------------------*'.format(label_name))
        logger.info('  *--------------------------------------------------*')

        # Get support sample + mask for current class.
        support_sample = test_dataset.getSupport(label=label_val, all_slices=args.all_slices, N=args.n_shot)
        test_dataset.label = label_val

        # Infer.
        with torch.no_grad():
            scores = infer(model, query_loader, support_sample, args, logger, label_name)

        # Log class-wise results
        class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
        class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()

        logger.info('      Mean class IoU: {}'.format(class_iou[label_name]))
        logger.info('      Mean class Dice: {}'.format(class_dice[label_name]))
        logger.info('  *--------------------------------------------------*')

    # Log final results.
    logger.info('  *-----------------Final results--------------------*')
    logger.info('  *--------------------------------------------------*')
    logger.info('  Mean IoU: {}'.format(class_iou))
    logger.info('  Mean Dice: {}'.format(class_dice))
    logger.info('  *--------------------------------------------------*')


def infer(model, query_loader, support_sample, args, logger, label_name):

    # Test mode.
    model.eval()

    # Unpack support data.
    support_image = [support_sample['image'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]  # n_shot x 3 x H x W
    support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]  # n_shot x H x W

    # Loop through query volumes.
    scores = Scores()
    for i, sample in enumerate(query_loader):

        # Unpack query data.
        query_image = [sample['image'][i].float().cuda() for i in range(sample['image'].shape[0])]  # [C x 3 x H x W]
        query_label = sample['label'].long()  # C x H x W
        query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]

        # Compute output.
        if args.EP1 is True:
            # Match support slice and query sub-chunck.
            query_pred = torch.zeros(query_label.shape[-3:])
            C_q = sample['image'].shape[1]
            idx_ = np.linspace(0, C_q, args.n_shot+1).astype('int')
            for sub_chunck in range(args.n_shot):
                support_image_s = [support_image[sub_chunck]]  # 1 x 3 x H x W
                support_fg_mask_s = [support_fg_mask[sub_chunck]]  # 1 x H x W
                query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck+1]]  # C' x 3 x H x W
                query_pred_s, _, _ = model([support_image_s], [support_fg_mask_s], [query_image_s], train=False)  # C x 2 x H x W
                query_pred_s = query_pred_s.argmax(dim=1).cpu()  # C x H x W
                query_pred[idx_[sub_chunck]:idx_[sub_chunck+1]] = query_pred_s

        else:  # EP 2
            query_pred, _, _ = model([support_image], [support_fg_mask], query_image, train=False)  # C x 2 x H x W
            query_pred = query_pred.argmax(dim=1).cpu()  # C x H x W

        # Record scores.
        scores.record(query_pred, query_label)

        # Log.
        logger.info('    Tested query volume: ' + sample['id'][0][len(args.data_root):]
                    + '. Dice score:  ' + str(scores.patient_dice[-1].item()))

        # Save predictions.
        file_name = 'image_' + query_id + '_' + label_name + '.pt'
        torch.save(query_pred, os.path.join(args.save, file_name))

    return scores


if __name__ == '__main__':
    main()
