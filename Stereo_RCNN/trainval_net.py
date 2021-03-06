# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick

# Modified by Peiliang Li for Stereo RCNN train
# --------------------------------------------------------

#-----------------------------------------------------------------------
# Adapted from https://github.com/bjarnekvae/Stereo-RCNN/tree/generalize
#-----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.stereo_rcnn.resnet import resnet
from torch.utils.data import Dataset
import cv2
import csv
import json
import torch.nn.functional as F


def parse_args():
    '''
  Parse input arguments
  '''
    parser = argparse.ArgumentParser(description='Train the Stereo R-CNN network')

    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20000, type=int)  # 1200000

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="/content/gdrive/MyDrive/Stereo_RCNN/models_stereo",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=8, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)

    # config optimization
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=10, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=6477, type=int)

    args = parser.parse_args()
    return args



class LarvaeDataset(Dataset):
    def __init__(self, file_path):  # , transform, convert, im_size=(1848, 3264)):
        print('Creating data set...\n')
        self.im_size = (642, 2560)  # (1848, 3264)
        # self.transform = transform
        # self.convert = convert
        self.file_path = file_path

        self.data = self.build_data_set()
        print("Amount of image pairs to train: ", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
    bbs = BoundingBoxesOnImage([BoundingBox(x1=self.Y[index, 0],
                                            y1=self.Y[index, 1],
                                            x2=self.Y[index, 2],
                                            y2=self.Y[index, 3])],
                               shape=(self.X[index].shape[0], self.X[index].shape[1], self.X[index].shape[2]))
    x, bbs_aug = self.transform(image=self.X[index], bounding_boxes=bbs)
    im_height = np.float(x.shape[0])
    im_width = np.float(x.shape[1])
    x = self.convert(x)
    bbs = bbs_aug.bounding_boxes[0]

    Y_0 = np.float(bbs.x1) / im_width * 2.0 - 1.0
    Y_1 = np.float(bbs.y1) / im_height * 2.0 - 1.0
    Y_2 = np.float(bbs.x2) / im_width * 2.0 - 1.0
    Y_3 = np.float(bbs.y2) / im_height * 2.0 - 1.0

    return x, np.array([Y_0, Y_1, Y_2, Y_3], dtype=np.float)
    '''
        return self.data[index]

    def build_data_set(self):
        data = list()

        with open(self.file_path + "labels.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')


            for i, row in enumerate(csv_reader):
                if row[5] != '{}' and i > 0:
                    attributes_dict = json.loads(row[5])
                    if attributes_dict['name'] == 'roi':
                        print(row[0])

                        img = cv2.imread(self.file_path + row[0])


                        l_img = img[:, :int(img.shape[1] / 2), :]
                        r_img = img[:, int(img.shape[1] / 2) - 1:-1, :]



                        # ROIS
                        l_rois = np.array(attributes_dict['left_rois'])
                        r_rois = np.array(attributes_dict['right_rois'])



                        data_point = self.compose_data(l_img, r_img, l_rois, r_rois)
                        data.append(data_point)


        return data

    def compose_data(self, l_img, r_img, l_rois, r_rois):
        data_point = list()

        # left and right images
        im_shape = l_img.shape
        im_size_min = np.min(im_shape[0:2])
        im_scale = float(cfg.TRAIN.SCALES[0]) / float(im_size_min)
        l_img = cv2.resize(l_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        r_img = cv2.resize(r_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        l_img = l_img.astype(np.float32, copy=False)
        r_img = r_img.astype(np.float32, copy=False)
        l_img -= cfg.PIXEL_MEANS
        r_img -= cfg.PIXEL_MEANS

        info = np.array([l_img.shape[0], l_img.shape[1], 1.0], dtype=np.float32)

        l_img = np.moveaxis(l_img.copy(), -1, 0)
        r_img = np.moveaxis(r_img.copy(), -1, 0)

        data_point.append(l_img)
        data_point.append(r_img)

        # Image info
        data_point.append(info)

        # left and right ROIS
        l_temp = np.zeros([30, 5])
        l_rois[:, 2] = l_rois[:, 0] + l_rois[:, 2]
        l_rois[:, 3] = l_rois[:, 1] + l_rois[:, 3]
        l_rois = l_rois * im_scale
        l_temp[0:l_rois.shape[0], 0:4] = l_rois
        l_temp[0:l_rois.shape[0], 4] = 1

        r_temp = np.zeros([30, 5])
        r_rois[:, 2] = r_rois[:, 0] + r_rois[:, 2]
        r_rois[:, 3] = r_rois[:, 1] + r_rois[:, 3]
        r_rois = r_rois * im_scale
        r_temp[0:r_rois.shape[0], 0:4] = r_rois
        r_temp[0:r_rois.shape[0], 4] = 1

        data_point.append(l_temp.copy())
        data_point.append(r_temp.copy())

        # Merged ROIS
        merge = np.zeros([30, 5])
        for i in range(30):
            merge[i, 0] = np.min([l_temp[i, 0], r_temp[i, 0]])
            merge[i, 1] = np.min([l_temp[i, 1], r_temp[i, 1]])
            merge[i, 2] = np.max([l_temp[i, 2], r_temp[i, 2]])
            merge[i, 3] = np.max([l_temp[i, 3], r_temp[i, 3]])

        merge[0:r_rois.shape[0], 4] = 1
        data_point.append(merge.copy())

        data_point.append(np.zeros([30, 5]))  # Object dimension (3) and viewpoint angle (1), orientation (1)? (?)
        data_point.append(np.zeros([30, 6]))  # Object sparse key points (?)
        data_point.append(r_rois.shape[0])  # Num objects

        return data_point.copy()


if __name__ == '__main__':
    args = parse_args()

    print('Using config:')
    np.random.seed(cfg.RNG_SEED)

    # imdb, roidb, ratio_list, ratio_index = combined_roidb('kitti_train')
    train_size = 1000  # len(roidb)



    output_dir = args.save_dir + '/'
    if not os.path.exists(output_dir):
        print('save dir', output_dir)
        os.makedirs(output_dir)
    log_info = open((output_dir + 'trainlog.txt'), 'w')


    def log_string(out_str):
        log_info.write(out_str + '\n')
        log_info.flush()
        print(out_str)


    train_dir = "/content/gdrive/MyDrive/Stereo_RCNN/data/training_data/"  # dataset path

    dataset = LarvaeDataset(
        file_path=train_dir)

    # Dataloader iterators
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)


    # initilize the tensor holder here.
    im_left_data = Variable(torch.FloatTensor(1).cuda())
    im_right_data = Variable(torch.FloatTensor(1).cuda())
    im_info = Variable(torch.FloatTensor(1).cuda())
    num_boxes = Variable(torch.LongTensor(1).cuda())
    gt_boxes_left = Variable(torch.FloatTensor([1]).cuda())
    gt_boxes_right = Variable(torch.FloatTensor(1).cuda())
    gt_boxes_merge = Variable(torch.FloatTensor(1).cuda())
    gt_dim_orien = Variable(torch.FloatTensor(1).cuda())
    gt_kpts = Variable(torch.FloatTensor(1).cuda())

    classes = ('__background__', 'Car')

    # initilize the network here.
    stereoRCNN = resnet(classes, 101, pretrained=True)

    stereoRCNN.create_architecture()

    lr = 0.0001

    uncert = Variable(torch.rand(6).cuda(), requires_grad=True)
    torch.nn.init.constant(uncert, -1.0)

    params = []
    for key, value in dict(stereoRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    params += [{'params': [uncert], 'lr': lr}]


    optimizer = torch.optim.Adam(params, lr=lr)


    if False:
        load_name = os.path.join(output_dir, "stereo_rcnn_epoch_3400_loss_-65.22175598144531.pth")
        log_string('loading checkpoint %s' % (load_name))
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        stereoRCNN.load_state_dict(checkpoint['model'])
        uncert.data = checkpoint['uncert']
        log_string('loaded checkpoint %s' % (load_name))

    stereoRCNN.cuda()



    iters_per_epoch = int(train_size / args.batch_size)
    for epoch in range(0, args.max_epochs + 1):

        stereoRCNN.train()
        start = time.time()



        data_iter = iter(dataloader)
        for step, data in enumerate(dataloader):
            # data = next(data_iter)
            im_left_data.resize_(data[0].size()).copy_(data[0])
            im_right_data.resize_(data[1].size()).copy_(data[1])
            im_info.resize_(data[2].size()).copy_(data[2])

            gt_boxes_left.resize_(data[3].size()).copy_(data[3])
            gt_boxes_right.resize_(data[4].size()).copy_(data[4])


            gt_boxes_merge.resize_(data[5].size()).copy_(data[5])
            gt_dim_orien.resize_(data[6].size()).copy_(data[6])
            gt_kpts.resize_(data[7].size()).copy_(data[7])
            num_boxes.resize_(data[8].size()).copy_(data[8])

            start = time.time()
            stereoRCNN.zero_grad()
            rois_left, rois_right, cls_prob, bbox_pred, dim_orien_pred, kpts_prob, \
            left_border_prob, right_border_prob, rpn_loss_cls, rpn_loss_box_left_right, \
            RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label = \
                stereoRCNN(im_left_data, im_right_data, im_info, gt_boxes_left, gt_boxes_right,
                           gt_boxes_merge, gt_dim_orien, gt_kpts, num_boxes)





            loss = rpn_loss_cls.mean() * torch.exp(-uncert[0]) + uncert[0] + \
                   rpn_loss_box_left_right.mean() * torch.exp(-uncert[1]) + uncert[1] + \
                   RCNN_loss_cls.mean() * torch.exp(-uncert[2]) + uncert[2] + \
                   RCNN_loss_bbox.mean() * torch.exp(-uncert[3]) + uncert[3] + \
                   RCNN_loss_dim_orien.mean() * torch.exp(-uncert[4]) + uncert[4] + \
                   RCNN_loss_kpts.mean() * torch.exp(-uncert[5]) + uncert[5]



            uncert_data = uncert.data


            optimizer.zero_grad()
            loss.backward()
            clip_gradient(stereoRCNN, 10.)



            optimizer.step()

            end = time.time()

            loss_rpn_cls = rpn_loss_cls.item()
            loss_rpn_box_left_right = rpn_loss_box_left_right.item()
            loss_rcnn_cls = RCNN_loss_cls.item()
            loss_rcnn_box = RCNN_loss_bbox.item()
            loss_rcnn_dim_orien = RCNN_loss_dim_orien.item()
            loss_rcnn_kpts = RCNN_loss_kpts
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt

            log_string(
                '[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e' % (epoch, step, iters_per_epoch, loss.item(), lr))
            log_string('\t\t\tfg/bg=(%d/%d), time cost: %f' % (fg_cnt, bg_cnt, end - start))
            log_string(
                '\t\t\trpn_cls: %.4f, rpn_box_left_right: %.4f, rcnn_cls: %.4f, rcnn_box_left_right %.4f,dim_orien %.4f, kpts %.4f' \
                % (loss_rpn_cls, loss_rpn_box_left_right, loss_rcnn_cls, loss_rcnn_box, loss_rcnn_dim_orien,
                   loss_rcnn_kpts))

            del rpn_loss_cls, rpn_loss_box_left_right, RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts

        if epoch % 10 == 0:
            save_name = os.path.join(output_dir, 'stereo_rcnn_epoch_{}_loss_{}.pth'.format(epoch, loss.item()))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': stereoRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'uncert': uncert.data,
            }, save_name)

            log_string('save model: {}'.format(save_name))
            end = time.time()
            log_string('time %.4f' % (end - start))






