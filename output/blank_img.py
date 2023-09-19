# read /home/linxi/workspace/POPE/data/minival2014/minival2014/COCO_val2014_000000000357.jpg
# then 0 multiple this image to create a blank image
# save
import cv2
import os
save_path = "../POPE/output/sub_adversarial_gt_obj"
img_file = "../POPE/data/minival2014/minival2014/COCO_val2014_000000009628.jpg"
img = cv2.imread(img_file)
print(img.shape)
img = img * 0
# save
cv2.imwrite(os.path.join(save_path, "blank.jpg"), img)
