from itertools import cycle
import numpy as np
import pandas as pd
import cv2

ALL_STEPS = ['left', 'down', 'right', 'up']
phi = 1.618

def get_subimage(img, mask):
    x = np.max(mask, 1).sum()
    y = np.max(mask, 0).sum()
    
    return np.moveaxis(np.stack(
        [np.extract(mask, img[:,:,0]).reshape(x, y)
        ,np.extract(mask, img[:,:,1]).reshape(x, y)
        ,np.extract(mask, img[:,:,2]).reshape(x, y)]
    ), 0, -1)

def clip_float(num, min_value=0, max_value=100):
    return min(max_value, max(min_value, num))

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

itten_color_wheel = [
 [53/360, 0.63, 0.94]
 ,[42/360, 0.51, 0.84]
 ,[32/360, 0.44, 0.97]
 ,[19/360, 0.40, 1.00 ]
 ,[349/360, 0.51, 0.40]
 ,[313/360, 0.32, 0.44]
 ,[268/360, 0.35, 0.68]
 ,[248/360, 0.49, 0.40]
 ,[233/360, 0.56, 0.64]
 ,[188/360, 0.32, 0.27]
 ,[130/360, 0.45, 0.22]
 ,[70/360, 0.49, 0.43]
]

other_ones = [
[240/460, 0.5, 0]
,[240/460, 0, 0]
]

np_itten_color_wheel = np.array(itten_color_wheel)
np_wb_itten_color_wheel = np.array(itten_color_wheel + other_ones)

def get_scaled_hls(img_hls):
    h, l, s = cv2.split(img_hls)
    h = np.ceil(360*(h/180))
    l = ((l/255))
    s = ((s/255))

    return np.stack([
        h, l, s
    ]).T

def get_full_normalized_hls(img_hls):
    h, l, s = cv2.split(img_hls)
    h = ((h/180))
    l = ((l/255))
    s = ((s/255))

    return np.stack([
        h, l, s
    ]).T

def get_subimage(img, mask):
    x = np.max(mask, 1).sum()
    y = np.max(mask, 0).sum()
    
    return np.moveaxis(np.stack(
        [np.extract(mask, img[:,:,0]).reshape(x, y)
        ,np.extract(mask, img[:,:,1]).reshape(x, y)
        ,np.extract(mask, img[:,:,2]).reshape(x, y)]
    ), 0, -1)

def get_color_wheel_quantized_img(fl_img_hls, color_wheel=np_itten_color_wheel):
    distance = cdist(fl_img_hls.reshape(-1,3), color_wheel).reshape(fl_img_hls.shape[0], fl_img_hls.shape[1], color_wheel.shape[0])

    itten_img = np.argmin(distance, axis=2)
        
    return itten_img

def get_kmeans_quantized_img(img_hls, clustering_method = KMeans()):
    km = KMeans()
    km = km.fit(get_full_normalized_hls(img_hls).reshape(-1,3))
    
    palette = km.cluster_centers_
    prediction = km.predict(get_full_normalized_hls(img_hls).reshape(-1,3)).reshape(img_hls.shape[1], img_hls.shape[0]).T
    
    return prediction, palette

def get_luminosity_quantized_img(img_hls, n_components=3):
    gmm = GaussianMixture(n_components=n_components)
    preds = gmm.fit_predict(img_hls[:,:,1].reshape(-1, 1))
    sorted_means = dict(zip(range(len(gmm.means_)), np.argsort(gmm.means_.reshape(-1))))
    
    pd_preds = pd.Series(preds).map(sorted_means)
    
    return pd_preds.values.reshape(img_hls.shape[0], img_hls.shape[1])

def get_rule_of_thirds_mask(img,n=3):
    original_x_size, original_y_size = img.shape[:2]
    
    mask_matrix = np.zeros(img.shape[:2])
    
    cnt = 0
    x_steps = list(range(0, original_x_size, int(original_x_size/n))) + [original_x_size]
    y_steps = list(range(0, original_y_size, int(original_y_size/n))) + [original_y_size]
    for from_x, to_x in list(zip(x_steps, x_steps[1:]))[:n]:
        for from_y, to_y in list(zip(y_steps, y_steps[1:]))[:n]:            
            mask_matrix[from_x:to_x, from_y:to_y] = cnt
            cnt += 1
    
    return mask_matrix

def get_golden_ratio_mask(img, side_size=100, anchor_point=(500, 500), start_step='left'):
    start_idx = ALL_STEPS.index(start_step)
    steps = ALL_STEPS[start_idx:] + ALL_STEPS[:start_idx]
    cycle_steps = cycle(steps)

    mask_img = np.zeros(img.shape[:2])

    b = side_size
    for idx, step in enumerate(cycle_steps):
        a = b 
        b = int(a*phi) 

        if step == 'left':
            anchor_point = anchor_point[0], anchor_point[1] - b
        elif step == 'down':
            anchor_point = anchor_point[0] + a, anchor_point[1]
        elif step == 'right':
            anchor_point = anchor_point[0] - (b - a), anchor_point[1] + a
        else:
            anchor_point = anchor_point[0] - b, anchor_point[1]  - (b - a)

        from_ud, to_ud = anchor_point[0], anchor_point[0] + b
        from_lr, to_lr = anchor_point[1], anchor_point[1] + b
            
        if (from_ud > img.shape[0] or from_ud < 0) and\
            (to_ud > img.shape[0] or to_ud < 0) and\
            (from_lr > img.shape[1] or from_lr < 0) and\
            (to_lr > img.shape[1] or to_lr < 0):
            break

        mask_img[clip_float(from_ud, max_value=img.shape[0]):clip_float(to_ud, max_value=img.shape[0]), clip_float(from_lr, max_value=img.shape[1]):clip_float(to_lr, max_value=img.shape[1])] = idx
        
    return mask_img