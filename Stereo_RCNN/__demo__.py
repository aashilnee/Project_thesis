# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick

# Modified by Peiliang Li for Stereo RCNN demo
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import time
import cv2
import torch
from torch.autograd import Variable
from model.utils.config import cfg
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, kpts_transform_inv, border_transform_inv
from model.stereo_rcnn.resnet import resnet
import glob
from tqdm import tqdm
from norfair import Detection, Tracker, Color, centroid

from stereo_manager import StereoManager, CameraParameters, StereoParameters

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test the Stereo R-CNN network')

  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models_stereo",
                      type=str)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=12, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=6477, type=int)

  args = parser.parse_args()
  return args

if __name__ == '__main__':

  args = parse_args()

  np.random.seed(cfg.RNG_SEED)

  # stereo manager

  sm = StereoManager()
  sm.load_calibration("/mnt/raid/digiras_302004685/calibration/calibration_params/1_stereo_calibration.pickle")


  def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

  tracker = Tracker(distance_function=euclidean_distance, distance_threshold=200) # TODO
  positions_list = []  # TODO

  # TODO
  fourcc = cv2.VideoWriter_fourcc(*'MP4V') # TODO
  # TODO
  out = cv2.VideoWriter("/mnt/raid/digiras_302004685/neural_net/analysed/video.mp4", fourcc, 25.0, (3264, 924))

  input_dir = args.load_dir + "/"
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  #load_name = os.path.join(input_dir,
  #  'stereo_rcnn_{}_{}.pth'.format(args.checkepoch, args.checkpoint))
  kitti_classes = np.asarray(['__background__', 'Car'])

  load_name = '/home/bjarne/raid/digiras_302004685/neural_net/models_uv/stereo_rcnn_epoch_3100_loss_-67.12100982666016.pth'

  # initilize the network here.
  stereoRCNN = resnet(kitti_classes, 101, pretrained=False)
  stereoRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  stereoRCNN.load_state_dict(checkpoint['model'])
  print('load model successfully!')

  cv2.namedWindow("img", cv2.WINDOW_NORMAL)


  with torch.no_grad():
    # initilize the tensor holder here.
    im_left_data = Variable(torch.FloatTensor(1).cuda())
    im_right_data = Variable(torch.FloatTensor(1).cuda())
    im_info = Variable(torch.FloatTensor(1).cuda())
    num_boxes = Variable(torch.LongTensor(1).cuda())
    gt_boxes = Variable(torch.FloatTensor(1).cuda())

    stereoRCNN.cuda()

    eval_thresh = 0.00
    vis_thresh = 0.01

    stereoRCNN.eval()

    # read data
    video = '/mnt/raid/digiras_302004685/data/200316_NFFT/kar2/D20210315T170450.mp4'

    cap = cv2.VideoCapture(video)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for frame_num in tqdm(range(4500, 5500)):
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
      ret, img = cap.read()
      im_shape = img.shape

      y_offset_value = 144

      r_img = img[:int(im_shape[0]/2), :, :]
      l_img = img[int(im_shape[0]/2)-1:-1, :, :]

      vis_img_l = l_img.copy()
      vis_img_r = r_img.copy()

      img_left = l_img[:-y_offset_value, :, :]
      img_right = r_img[y_offset_value:, :, :]

      im_shape = img_left.shape
      im_size_min = np.min(im_shape[0:2])
      im_scale = float(cfg.TRAIN.SCALES[0]) / float(im_size_min)
      img_left = cv2.resize(img_left, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
      img_right = cv2.resize(img_right, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

      info = np.array([[img_left.shape[0], img_left.shape[1], im_scale]], dtype=np.float32)

      img_left = img_left.astype(np.float32, copy=False)
      img_right = img_right.astype(np.float32, copy=False)

      img_left -= cfg.PIXEL_MEANS
      img_right -= cfg.PIXEL_MEANS

      img_left = torch.from_numpy(img_left)
      img_left = img_left.permute(2, 0, 1).unsqueeze(0).contiguous()

      img_right = torch.from_numpy(img_right)
      img_right = img_right.permute(2, 0, 1).unsqueeze(0).contiguous()

      info = torch.from_numpy(info)

      im_left_data.resize_(img_left.size()).copy_(img_left)
      im_right_data.resize_(img_right.size()).copy_(img_right)
      im_info.resize_(info.size()).copy_(info)

      det_tic = time.time()
      rois_left, rois_right, cls_prob, bbox_pred, bbox_pred_dim, kpts_prob,\
      left_prob, right_prob, rpn_loss_cls, rpn_loss_box_left_right,\
      RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label =\
      stereoRCNN(im_left_data, im_right_data, im_info, gt_boxes, gt_boxes, gt_boxes, gt_boxes, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes_left = rois_left.data[:, :, 1:5]
      boxes_right = rois_right.data[:, :, 1:5]


      bbox_pred = bbox_pred.data
      box_delta_left = bbox_pred.new(bbox_pred.size()[1], 4*len(kitti_classes)).zero_()
      box_delta_right = bbox_pred.new(bbox_pred.size()[1], 4*len(kitti_classes)).zero_()

      for keep_inx in range(box_delta_left.size()[0]):
        box_delta_left[keep_inx, 0::4] = bbox_pred[0,keep_inx,0::6]
        box_delta_left[keep_inx, 1::4] = bbox_pred[0,keep_inx,1::6]
        box_delta_left[keep_inx, 2::4] = bbox_pred[0,keep_inx,2::6]
        box_delta_left[keep_inx, 3::4] = bbox_pred[0,keep_inx,3::6]

        box_delta_right[keep_inx, 0::4] = bbox_pred[0,keep_inx,4::6]
        box_delta_right[keep_inx, 1::4] = bbox_pred[0,keep_inx,1::6]
        box_delta_right[keep_inx, 2::4] = bbox_pred[0,keep_inx,5::6]
        box_delta_right[keep_inx, 3::4] = bbox_pred[0,keep_inx,3::6]

      box_delta_left = box_delta_left.view(-1,4)
      box_delta_right = box_delta_right.view(-1,4)


      box_delta_left = box_delta_left * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
      box_delta_right = box_delta_right * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()


      box_delta_left = box_delta_left.view(1,-1,4*len(kitti_classes))
      box_delta_right = box_delta_right.view(1, -1,4*len(kitti_classes))

      pred_boxes_left = bbox_transform_inv(boxes_left, box_delta_left, 1)
      pred_boxes_right = bbox_transform_inv(boxes_right, box_delta_right, 1)

      pred_boxes_left = clip_boxes(pred_boxes_left, im_info.data, 1)
      pred_boxes_right = clip_boxes(pred_boxes_right, im_info.data, 1)

      pred_boxes_left /= im_info[0, 2].data
      pred_boxes_right /= im_info[0, 2].data

      scores = scores.squeeze()[:,1]
      pred_boxes_left = pred_boxes_left.squeeze()
      pred_boxes_right = pred_boxes_right.squeeze()

      det_toc = time.time()
      detect_time = det_toc - det_tic

      inds = torch.nonzero(scores > eval_thresh).view(-1)

      det_l = np.zeros([0, 2], dtype=np.int)
      det_r = np.zeros([0, 2], dtype=np.int)
      det_3d = np.zeros([0, 3], dtype=np.int)

      if inds.numel() > 0:
        cls_scores = scores[inds]
        _, order = torch.sort(cls_scores, 0, True)

        cls_boxes_left = pred_boxes_left[inds][:, 4:8]
        cls_boxes_right = pred_boxes_right[inds][:, 4:8]

        cls_dets_left = torch.cat((cls_boxes_left, cls_scores.unsqueeze(1)), 1)
        cls_dets_right = torch.cat((cls_boxes_right, cls_scores.unsqueeze(1)), 1)

        cls_dets_left = cls_dets_left[order]
        cls_dets_right = cls_dets_right[order]

        keep = nms(cls_boxes_left[order, :], cls_scores[order], cfg.TEST.NMS)
        keep = keep.view(-1).long()
        cls_dets_left = cls_dets_left[keep]
        cls_dets_right = cls_dets_right[keep]

        #im2show_left = vis_detections(im2show_left, kitti_classes[j], cls_dets_left.cpu().numpy(), vis_thresh)
        #im2show_right = vis_detections(im2show_right, kitti_classes[j], cls_dets_right.cpu().numpy(), vis_thresh)

        l_rois = cls_dets_left.cpu().numpy()
        r_rois = cls_dets_right.cpu().numpy()

        for i, roi in enumerate(l_rois):
          color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
          r_score = r_rois[i, -1]
          l_score = l_rois[i, -1]
          if l_score > vis_thresh and r_score > vis_thresh:
            l_bbox = tuple(int(np.round(x)) for x in l_rois[i, :4])
            r_bbox = tuple(int(np.round(x)) for x in r_rois[i, :4])
            #vis_img_l = cv2.rectangle(vis_img_l, l_bbox[0:2], l_bbox[2:4], color, 5)
            #vis_img_r = cv2.rectangle(vis_img_r, r_bbox[0:2], r_bbox[2:4], color, 5)

            left_key = np.array([l_bbox[0] + int((l_bbox[2] - l_bbox[0])/2), l_bbox[1] + int((l_bbox[3] - l_bbox[1])/2)], dtype=np.int)
            right_key = np.array([r_bbox[0] + int((r_bbox[2] - r_bbox[0])/2), r_bbox[1] + int((r_bbox[3] - r_bbox[1])/2) + y_offset_value], dtype=np.int)
            det_l = np.vstack((det_l, left_key))
            det_r = np.vstack((det_r, right_key))

            sl_key = np.array([left_key], dtype=np.float32)
            sr_key = np.array([right_key], dtype=np.float32)
            xyz = sm.stereopixel_to_real(sl_key, sr_key)
            det_3d = np.vstack((det_3d, xyz))


        detected_fish = []
        for i in range(det_l.shape[0]):
          detected_fish.append(Detection(det_l[i], data=det_3d))

        tracked_objects = tracker.update(detections=detected_fish)

        i = 0
        est_l_keys = []
        for obj in tracked_objects:
          if not obj.live_points.any():
            continue

          l_xy = np.squeeze(obj.estimate).astype(int)
          #col = Color.random(obj.id)

          est_l_keys.append([l_xy, obj.id])

        for i in range(det_l.shape[0]):    # TODO
          col = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
          #vis_img_l = cv2.circle(vis_img_l, tuple(det_l[i]), 50, col, 5)
          vis_img_r = cv2.circle(vis_img_r, tuple(det_r[i]), 50, col, 10)

          for est_l_key, est_id in est_l_keys:
            if np.abs(est_l_key[0] - det_l[i, 0]) < 100 and np.abs(est_l_key[1] - det_l[i, 1]) < 100:
              positions_list.append([est_l_key, det_3d[i, :], est_id])

              if len(positions_list) > 50:
                positions_list.pop(0)

        for i, positions in enumerate(positions_list):
          est_l_key = positions[0]
          det_3d = positions[1]
          est_id = positions[2]

          if i == len(positions_list)-1:
            dot_size = int((600.0 - det_3d[2]) / 200.0 * 50)
            if dot_size < 10:
              dot_size = 10
            vis_img_l = cv2.circle(vis_img_l, tuple(est_l_key), dot_size, Color.random(est_id), 10)
            ims_str = "X: {}\nY: {}\nZ: {}".format(int(det_3d[0]), int(det_3d[1]), int(det_3d[2]))
            y0, dy = est_l_key[1] - 100, 30
            for j, line in enumerate(ims_str.split('\n')):
              y = y0 + j * dy
              vis_img_l = cv2.putText(vis_img_l, line, (est_l_key[0] - 150, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                      2,
                                      cv2.LINE_AA)
          else:
            dot_size = int((600.0-det_3d[2]) / 200.0 * 10)
            if dot_size < 3:
              dot_size = 3
            vis_img_l = cv2.circle(vis_img_l, tuple(est_l_key), dot_size, Color.random(est_id), -1)

        img = np.hstack((vis_img_l, vis_img_r))
        im_scale = 0.5
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("img", img)
        cv2.waitKey()

        #img_name = "frame_" + str(frame_num) + '.jpg'
        #cv2.imwrite("/mnt/raid/digiras_302004685/neural_net/analysed/" + img_name, img)
        out.write(img)

  cap.release()
  out.release()