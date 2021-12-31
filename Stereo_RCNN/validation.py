import cv2
import numpy as np
import os


green = (87,201,0)
red = (0,0,128)
blue = (128,0,0)

#TODO--------------------------------------- BILDE 1 (200) ----------------------------------------------------------------
'''
# {"name": "roi", "left_rois": [[168, 31, 464, 114], [140, 188, 353, 83], [228, 280, 368, 80], [828, 339, 349, 99]], "right_rois": [[232, 31, 464, 114], [181, 188, 353, 83], [277, 280, 368, 80], [864, 339, 349, 99]]}

img_l_path = 'demo/left_200.png' # TODO
img_r_path = 'demo/right_200.png' # TODO

img_left = cv2.imread(img_l_path)
img_right = cv2.imread(img_r_path)

im2show_left = np.copy(cv2.imread(img_l_path))
im2show_right = np.copy(cv2.imread(img_r_path))

left_predicted_1 = [(162, 42, 624, 150), (223, 278, 602, 358), (830, 338, 1189, 426), (111, 190, 496, 275)]
right_predicted_1 = [(224, 42, 687, 150), (265, 278, 644, 358), (857, 338, 1215, 426), (133, 190, 515, 275)]

left_true_1 = [(168, 31, 464+168, 114+31), (140, 188, 353+140, 83+188), (228, 280, 368+228, 80+280), (828, 339, 349+828, 99+330), (427, 357, 367+427, 91+349)]     # (828, 339, 349+828, 99+339)]
right_true_1 = [(232, 31, 464+232, 114+31), (181, 188, 353+181, 83+188), (277, 280, 368+277, 80+280), (864, 339, 349+864, 99+330), (469, 357, 367+469, 91+349)]    # (864, 339, 349+864, 99+339)]

for i, roi in enumerate(left_true_1):
    left_true = left_true_1[i]
    right_true = right_true_1[i]

    im2show_left = cv2.rectangle(im2show_left, left_true[0:2], left_true[2:4], green, 2)
    im2show_right = cv2.rectangle(im2show_right, right_true[0:2], right_true[2:4], green, 2)

for i, roi in enumerate(left_predicted_1):
    left_pred = left_predicted_1[i]
    right_pred = right_predicted_1[i]

    im2show_left = cv2.rectangle(im2show_left, left_pred[0:2], left_pred[2:4], red, 2)
    im2show_right = cv2.rectangle(im2show_right, right_pred[0:2], right_pred[2:4], red, 2)



# TODO når den er inni forloopen blir bare ett og ett bilde laget
img = np.hstack((im2show_left, im2show_right))
# Resize image
im_scale = 0.5
img = cv2.resize(img, None, None, fx=0.6, fy=0.8, interpolation=cv2.INTER_LINEAR)

cv2.imshow("img", img)
cv2.waitKey()

# Save image
#path = 'results/'
#cv2.imwrite(os.path.join(path, '200_val.jpg'), img)

# Adapted from https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef
def IoU(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union
    return iou



pred_box = [857, 338, 1215, 426]
true_box = [864, 339, 349+864, 99+330]



iou = IoU(true_box, pred_box)

print(iou)

'''

#TODO--------------------------------------- BILDE 2 (292) ----------------------------------------------------------------

# {"name": "roi", "left_rois": [[197, 30, 466, 113], [95, 196, 354, 73], [268, 275, 371, 80], [822, 348, 345, 88], [423, 357, 367, 91]],
# "right_rois": [[266, 30, 466, 113], [142, 196, 354, 73], [317, 275, 371, 80], [861, 348, 345, 88], [465, 357, 367, 91]]}
'''
img_l_path = 'demo/left_292.png' # TODO
img_r_path = 'demo/right_292.png' # TODO

img_left = cv2.imread(img_l_path)
img_right = cv2.imread(img_r_path)

im2show_left = np.copy(cv2.imread(img_l_path))
im2show_right = np.copy(cv2.imread(img_r_path))


left_predicted_2 = [(823, 348, 1170, 435), (206, 29, 662, 137), (263, 268, 648, 354), (89, 195, 444, 275) ]
right_predicted_2 = [(858, 348, 1206, 435), (239, 29, 695, 137), (294, 268, 679, 354), (133, 195, 487, 275)]

left_true_2 = [(197, 30, 466+197, 113+30), (95, 196, 354+95, 73+196), (268, 275, 371+268, 80+275), (822, 348, 345+822, 88+348), (423, 357, 367+423, 91+357)]
right_true_2 = [(266, 30, 466+266, 113+30), (142, 196, 354+142, 73+196), (317, 275, 371+317, 80+275), (861, 348, 345+861, 88+348), (465, 357, 367+465, 91+357)]

for i, roi in enumerate(left_true_2):

    left_true = left_true_2[i]
    right_true = right_true_2[i]

    im2show_left = cv2.rectangle(im2show_left, left_true[0:2], left_true[2:4], green, 2)
    im2show_right = cv2.rectangle(im2show_right, right_true[0:2], right_true[2:4], green, 2)

for i, roi in enumerate(left_predicted_2):
    left_pred = left_predicted_2[i]
    right_pred = right_predicted_2[i]

    im2show_left = cv2.rectangle(im2show_left, left_pred[0:2], left_pred[2:4], red, 2)
    im2show_right = cv2.rectangle(im2show_right, right_pred[0:2], right_pred[2:4], red, 2)



# TODO når den er inni forloopen blir bare ett og ett bilde laget
img = np.hstack((im2show_left, im2show_right))
# Resize image
im_scale = 0.5
img = cv2.resize(img, None, None, fx=0.6, fy=0.8, interpolation=cv2.INTER_LINEAR)

cv2.imshow("img", img)
cv2.waitKey()

# Save image
path = 'results/'
cv2.imwrite(os.path.join(path, '292_val.jpg'), img)

'''

#TODO--------------------------------------- BILDE 3 (338) ----------------------------------------------------------------
'''
# {"name": "roi", "left_rois": [[230, 28, 452, 127], [75, 179, 343, 77], [270, 282, 366, 75], [822, 348, 345, 88], [408, 362, 378, 79], [426, 206, 363, 72]],
# "right_rois": [[302, 28, 452, 127], [122, 179, 343, 77], [317, 282, 366, 75], [861, 348, 345, 88], [462, 362, 378, 79], [473, 206, 363, 72]]}

img_l_path = 'demo/left_338.png' # TODO
img_r_path = 'demo/right_338.png' # TODO

img_left = cv2.imread(img_l_path)
img_right = cv2.imread(img_r_path)

im2show_left = np.copy(cv2.imread(img_l_path))
im2show_right = np.copy(cv2.imread(img_r_path))


left_predicted_3 = [(227, 36, 674, 140), (816, 345, 1170, 434), (264, 276, 638, 366), (417, 206, 788, 279), (103, 182, 434, 254)]
right_predicted_3 = [(283, 36, 729, 140), (854, 345, 1206, 434), (313, 276, 687, 366), (462, 206, 835, 279), (143, 182, 475, 254)]

left_true_3 = [(230, 28, 452+230, 127+28), (75, 179, 343+75, 77+179), (270, 282, 366+270, 75+282), (822, 348, 345+822, 88+348), (408, 362, 378+408, 79+362), (426, 206, 363+426, 72+206)]
right_true_3 = [(302, 28, 452+302, 127+28), (122, 179, 343+122, 77+179), (317, 282, 366+317, 75+282), (861, 348, 345+861, 88+348), (462, 362, 378+462, 79+362), (473, 206, 363+473, 72+206)]


for i, roi in enumerate(left_true_3):

    left_true = left_true_3[i]
    right_true = right_true_3[i]

    im2show_left = cv2.rectangle(im2show_left, left_true[0:2], left_true[2:4], green, 2)
    im2show_right = cv2.rectangle(im2show_right, right_true[0:2], right_true[2:4], green, 2)

for i, roi in enumerate(left_predicted_3):
    left_pred = left_predicted_3[i]
    right_pred = right_predicted_3[i]

    im2show_left = cv2.rectangle(im2show_left, left_pred[0:2], left_pred[2:4], red, 2)
    im2show_right = cv2.rectangle(im2show_right, right_pred[0:2], right_pred[2:4], red, 2)



# TODO når den er inni forloopen blir bare ett og ett bilde laget
img = np.hstack((im2show_left, im2show_right))
# Resize image
im_scale = 0.5
img = cv2.resize(img, None, None, fx=0.6, fy=0.8, interpolation=cv2.INTER_LINEAR)

cv2.imshow("img", img)
cv2.waitKey()

# Save image
path = 'results/'
cv2.imwrite(os.path.join(path, '338_val.jpg'), img)


'''

#TODO--------------------------------------- BILDE 4 (384) ----------------------------------------------------------------

# {"name": "roi", "left_rois": [[337, 60, 365, 108], [50, 176, 337, 79], [253, 291, 360, 76], [437, 367, 369, 79], [405, 216, 375, 71]],
# "right_rois": [[428, 60, 365, 108], [84, 176, 337, 79], [296, 291, 360, 76], [481, 367, 369, 79], [443, 216, 375, 71]]}




img_l_path = 'demo/left_384.png' # TODO
img_r_path = 'demo/right_384.png' # TODO

img_left = cv2.imread(img_l_path)
img_right = cv2.imread(img_r_path)

im2show_left = np.copy(cv2.imread(img_l_path))
im2show_right = np.copy(cv2.imread(img_r_path))

left_predicted_4 = [(330, 62, 715, 160), (249, 290, 620, 371), (57, 179, 400, 257)]
right_predicted_4 = [(394, 62, 778, 160), (290, 290, 661, 371), (104, 179, 446, 257)]

left_true_4 = [(337, 60, 365+337, 108+60), (50, 176, 337+50, 79+176), (253, 291, 360+253, 76+291), (437, 367, 369+437, 79+367), (405, 216, 375+405, 71+216)]
right_true_4 = [(428, 60, 365+428, 108+60), (84, 176, 337+84, 79+176), (296, 291, 360+296, 76+291), (481, 367, 369+481, 79+367), (443, 216, 375+443, 71+216)]

for i, roi in enumerate(left_true_4):

    left_true = left_true_4[i]
    right_true = right_true_4[i]

    im2show_left = cv2.rectangle(im2show_left, left_true[0:2], left_true[2:4], green, 2)
    im2show_right = cv2.rectangle(im2show_right, right_true[0:2], right_true[2:4], green, 2)

for i, roi in enumerate(left_predicted_4):
    left_pred = left_predicted_4[i]
    right_pred = right_predicted_4[i]

    im2show_left = cv2.rectangle(im2show_left, left_pred[0:2], left_pred[2:4], red, 2)
    im2show_right = cv2.rectangle(im2show_right, right_pred[0:2], right_pred[2:4], red, 2)



# TODO når den er inni forloopen blir bare ett og ett bilde laget
img = np.hstack((im2show_left, im2show_right))
# Resize image
im_scale = 0.5
img = cv2.resize(img, None, None, fx=0.6, fy=0.8, interpolation=cv2.INTER_LINEAR)

cv2.imshow("img", img)
cv2.waitKey()

# Save image
path = 'results/'
cv2.imwrite(os.path.join(path, '384_val.jpg'), img)

def IoU(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union
    return iou



pred_box = [249, 290, 620, 371]
true_box = [253, 291, 360+253, 76+291]



iou = IoU(true_box, pred_box)

print(iou)
