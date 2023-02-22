import matplotlib.pyplot as plt
import numpy as np
import cv2


def det_vis(img, mask, n_classes=2, radius=3):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mask_values = [int(v / (n_classes - 1) * 255) for v in range(1, n_classes)] if n_classes > 1 else [255]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    for i, v in enumerate(mask_values):
        cnts, _ = cv2.findContours(np.uint8(mask == v), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area_true = np.pi * radius ** 2
            if cv2.contourArea(cnt) < area_true * 0.3 and radius > 5:
                continue
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            else:
                while isinstance(cnt[0], np.ndarray):
                    cnt = cnt[0]
                center = cnt
            cv2.circle(img, center, 3, colors[i], -1)
    return img


# def plot_img_and_mask(img, mask):
#     classes = mask.max() + 1
#     fig, ax = plt.subplots(1, classes)
#     ax[0].set_title('Input image')
#     ax[0].imshow(img)
#     ax[0].set_xticks([]), ax[0].set_yticks([])
#     for i in range(1, classes):
#         ax[i].set_title(f'Mask (class {i})')
#         ax[i].imshow(mask == i)
#         ax[i].set_xticks([]), ax[i].set_yticks([])
#     plt.show()


DATA_SET = {0: ("Fluo-N2DH-SIM+", 3),
            1: ("Fluo-C2DL-MSC", 3),
            2: ("Fluo-N2DH-GOWT1", 3),
            3: ("PhC-C2DL-PSC", 2),
            4: ("BF-C2DL-HSC", 3),
            5: ("Fluo-N2DL-HeLa", 3),
            6: ("BF-C2DL-MuSC", 3),
            7: ("DIC-C2DH-HeLa", 3),
            8: ("PhC-C2DH-U373", 15)}
