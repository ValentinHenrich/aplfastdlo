import os, cv2
import numpy as np
from unifastdlo.fastdlo.core_binary import Pipeline


def fastdlo(IMG_PATH, MASK_PATH):

    ######################
    #IMG_PATH = "EE_Pictures/Binaermaske02_original.jpg"
    #MASK_PATH = "EE_Pictures/Binaermaske02.jpg"
    ckpt_siam_name = "CP_similarity.pth"
    ckpt_seg_name = "CP_segmentation.pth"
    IMG_W = 640
    IMG_H = 360
    ######################

    script_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint_siam = os.path.join(script_path, "weights/" + ckpt_siam_name)
    checkpoint_seg = os.path.join(script_path, "weights/" + ckpt_seg_name)
    
    p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_W, img_h=IMG_H)

    # COLOR
    source_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    source_img = cv2.resize(source_img, (IMG_W, IMG_H))

    # MASK
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_COLOR)
    mask = cv2.resize(mask, (IMG_W, IMG_H))

    img_out, _ = p.run(source_img=source_img, mask_img=mask, mask_th=77)
    
    dlo_pixels = np.where(img_out > 5)
    if dlo_pixels[0].size > 0:
        return True # DLO-Pixels detected
    else:
        return False # No DLO-Pixels detected
"""
    cv2.imshow("img_out", img_out)
    canvas = source_img.copy()
    canvas = cv2.addWeighted(canvas, 1.0, img_out, 0.8, 0.0)
    cv2.imshow("output", canvas)
    cv2.waitKey(0)
"""