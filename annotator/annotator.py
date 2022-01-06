################################
# Adapted from Bjarne Kvæstad
################################
import cv2
import numpy as np
import os
import json
from tqdm import tqdm
from normxcorr2 import normxcorr2

rect_edge = 10
rect_adj = 30

data_path = '/Users/47900/Documents/videos/'
labeled_path = '/Users/47900/Documents/training_data/'
left_image_path = '/Users/47900/Documents/training_data/images2/'
right_image_path = '/Users/47900/Documents/training_data/images3/'

label_file = labeled_path + 'labels.csv'

if not os.path.exists(label_file):
    with open(label_file, 'a+') as f:
        f.write("filename;file_size;file_attributes;region_count;region_id;region_shape_attributes;region_attributes")

mode = 'anno'
ix, iy = -1, -1
l_rois = []
r_rois = []
change = False

# roi colors
colors = {0: (0, 128, 0),
          1: (255, 255, 255),
          2: (0, 128, 128),
          3: (0, 0, 0),
          4: (128, 0, 0),
          5: (0, 0, 255),
          6: (0, 0, 128),
          7: (128, 128, 128),
          8: (128, 0, 128),
          9: (0, 255, 255),
          10: (0, 255, 0),
          11: (255, 0, 255),
          12: (255, 255, 0),
          13: (255, 0, 0),
          14: (128, 128, 0),
          15: (192, 192, 192)}


# mouse callback function
def mark(event, x, y, flags, param):
    global ix, iy, mode, change, roi_id, roi_side

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        change = True
        for i in range(len(l_rois)):
            p2 = (l_rois[i][0] + l_rois[i][2], l_rois[i][1] + l_rois[i][3])

            if p2[0] - rect_adj < x < p2[0] + rect_adj and p2[1] - rect_adj < y < p2[1] + rect_adj:
                if mode == 'anno':
                    roi_id = i
                    mode = 'adj'
                return

        for i in range(len(l_rois)):
            l_p1 = (l_rois[i][0], l_rois[i][1])
            l_p2 = (l_rois[i][0] + l_rois[i][2], l_rois[i][1] + l_rois[i][3])
            r_p1 = (r_rois[i][0], r_rois[i][1])
            r_p2 = (r_rois[i][0] + r_rois[i][2], r_rois[i][1] + r_rois[i][3])

            if l_p1[0] < x < l_p2[0] and l_p1[1] < y < l_p2[1]:
                if mode == 'del':
                    l_rois.pop(i)
                    r_rois.pop(i)
                    change = False
                elif mode == 'anno':
                    roi_id = i
                    roi_side = 'l'
                    ix = x - l_rois[roi_id][0]
                    iy = y - l_rois[roi_id][1]
                    mode = 'move'
                return

            if r_p1[0] < x < r_p2[0] and r_p1[1] < y < r_p2[1]:
                if mode == 'del':
                    l_rois.pop(i)
                    r_rois.pop(i)
                    change = False
                elif mode == 'anno':
                    roi_id = i
                    roi_side = 'r'
                    ix = x - r_rois[roi_id][0]
                    iy = y - r_rois[roi_id][1]
                    mode = 'move'
                return

        if mode != 'del' and x < frame_width:
            l_rois.append([ix, iy, 0, 0])
            r_rois.append([frame_width + ix, iy, 0, 0])
        else:
            change = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if change:
            if mode == 'anno' and x < frame_width:
                start_x, start_y = ix, iy
                end_x, end_y = x, y
                l_rois[-1][2] = end_x - start_x  # width
                l_rois[-1][3] = end_y - start_y  # height

                start_x, start_y = r_rois[-1][0], r_rois[-1][1]
                end_x, end_y = frame_width + x, y
                r_rois[-1][2] = end_x - start_x  # width
                r_rois[-1][3] = end_y - start_y  # height
            elif mode == 'move':
                if roi_side == 'l':
                    l_rois[roi_id][0] = x - ix
                    l_rois[roi_id][1] = y - iy
                    r_rois[roi_id][1] = y - iy
                elif roi_side == 'r':
                    r_rois[roi_id][0] = x - ix
                    r_rois[roi_id][1] = y - iy
                    l_rois[roi_id][1] = y - iy
            elif mode == 'adj' and x < frame_width:
                w = x - l_rois[roi_id][0]
                h = y - l_rois[roi_id][1]
                if w < 0:
                    w = 0
                if h < 0:
                    h = 0

                l_rois[roi_id][2] = w
                l_rois[roi_id][3] = h

                r_rois[roi_id][2] = w
                r_rois[roi_id][3] = h

    elif event == cv2.EVENT_LBUTTONUP:
        if change:
            change = False
            if mode == 'move' or mode == 'adj':
                mode = 'anno'
                return
            if mode == 'anno':
                # roi
                l_start_x, l_start_y = l_rois[-1][0], l_rois[-1][1]
                l_end_x, l_end_y = x, y

                if l_end_x > l_start_x:
                    l_rois[-1][0] = l_start_x
                    l_rois[-1][2] = l_end_x - l_start_x  # width
                else:
                    l_rois[-1][0] = l_end_x
                    l_rois[-1][2] = l_start_x - l_end_x  # width

                if l_end_y > l_start_y:
                    l_rois[-1][1] = l_start_y
                    l_rois[-1][3] = l_end_y - l_start_y  # height
                else:
                    l_rois[-1][1] = l_end_y
                    l_rois[-1][3] = l_start_y - l_end_y  # height

                r_start_x, r_start_y = r_rois[-1][0], r_rois[-1][1]
                r_end_x, r_end_y = frame_width + x, y

                if r_end_x > r_start_x:
                    r_rois[-1][0] = r_start_x
                    r_rois[-1][2] = r_end_x - r_start_x  # width
                else:
                    r_rois[-1][0] = r_end_x
                    r_rois[-1][2] = r_start_x - r_end_x  # width

                if r_end_y > r_start_y:
                    r_rois[-1][1] = r_start_y
                    r_rois[-1][3] = r_end_y - r_start_y  # height
                else:
                    r_rois[-1][1] = r_end_y
                    r_rois[-1][3] = r_start_y - r_end_y  # height

                if l_rois[-1][2] < 10 or l_rois[-1][3] < 10:
                    l_rois.pop(-1)
                    r_rois.pop(-1)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', mark)

videos = [data_path + o for o in os.listdir(data_path) if o.endswith(('.mp4'))]  # , '.mkv'   '.mp4'
videos = [videos[0]]  # 8 videos (1 .mkv, 7 .mp4)

for video in videos:
    cap = cv2.VideoCapture(video)  # reading video file
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


    rect_edge = int(rect_edge * frame_width / 3264)
    rect_adj = int(rect_adj * frame_width / 3264)

    for frame_num in tqdm(range(200, int(frame_count), int(frame_count / 500))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, img = cap.read()
        im_shape = img.shape

        l_img = img[:int(im_shape[0] / 2), :, :]
        r_img = img[int(im_shape[0] / 2) - 1:-1, :, :]

        # gray-scale image used in cross-correlation
        right_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
        left_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)

        # cross-correlation
        corr = normxcorr2(left_img, right_img)
        # cross corr between equal image
        corr_1 = normxcorr2(left_img, left_img)

        # find match
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        # peak left image
        y_1, x_1 = np.unravel_index(np.argmax(corr_1), corr_1.shape)
        # offset
        y_offset_value = y_1 - y

        # when error in cross-correlation occur
        if y_offset_value < 70:
            print("\nError: Offset value too low. Offset value: ", y_offset_value)
            y_offset_value = 70
        elif y_offset_value > 78:
            print("\nError: Offset value too high. Offset value: ", y_offset_value)
            y_offset_value = 78
        else:
            print("\nOffset value: ", y_offset_value)

        l_img = l_img[:-y_offset_value, :, :]
        r_img = r_img[y_offset_value:, :, :]

        img = np.hstack((l_img, r_img))

        # crop images to equal sizes (2560x642)
        crop_value = 78 - y_offset_value
        if crop_value > 0:
            img = img[:-crop_value, :, :]
            print("\nCropped image size: ", img.shape)
        else:
            print("\nImage size: ", img.shape)

        while (1):
            vis_img = img.copy()
            for i, roi in enumerate(l_rois):
                if i > 15:
                    i = 15
                p1 = (roi[0], roi[1])
                p2 = (roi[0] + roi[2], roi[1] + roi[3])
                cv2.rectangle(vis_img, p1, p2, colors[i], rect_edge)
                cv2.rectangle(vis_img, (p2[0] - rect_adj, p2[1] - rect_adj), (p2[0] + rect_adj, p2[1] + rect_adj),
                              colors[i], rect_edge)

            for i, roi in enumerate(r_rois):
                if i > 15:
                    i = 15
                p1 = (roi[0], roi[1])
                p2 = (roi[0] + roi[2], roi[1] + roi[3])
                cv2.rectangle(vis_img, p1, p2, colors[i], rect_edge)

            cv2.imshow('image', vis_img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('d'):
                mode = 'del'
            if k == ord('a'):
                mode = 'anno'
            if k == ord('w'):
                if len(l_rois) == 0:
                    break

                for r_roi in r_rois:
                    r_roi[0] = r_roi[0] - frame_width

                # Create CSV
                region_shape_attributes = {"name": "roi",
                                           "left_rois": l_rois,
                                           "right_rois": r_rois}
                region_attributes = {"region": "body"}

                region_shape_attributes = json.dumps(region_shape_attributes)
                region_attributes = json.dumps(region_attributes)

                frame_name = video.split('/')[-1][:-4] + "_" + str(frame_num) + ".png"

                roiLeft = img[0:img.shape[0], 0:frame_width]
                roiRight = img[0:img.shape[0], frame_width:img.shape[1]]

                cv2.imwrite(left_image_path + frame_name, roiLeft)
                cv2.imwrite(right_image_path + frame_name, roiRight)

                size = os.path.getsize(left_image_path + frame_name)

                with open(label_file, 'a+') as f:
                    f.writelines("\n{};{};{};{};{};{};{}".format(frame_name, size, '{}', 1, 1,
                                                                 region_shape_attributes, region_attributes))


                for r_roi in r_rois:
                    r_roi[0] = r_roi[0] + frame_width

                break
            elif k == 27:
                # TODO save ROIS to file + exit
                cv2.destroyAllWindows()
                quit()



