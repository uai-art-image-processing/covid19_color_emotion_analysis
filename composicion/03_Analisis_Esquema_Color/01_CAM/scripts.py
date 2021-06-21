import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import seaborn as sns
import colorsys
import joblib

from tqdm import tqdm
from itertools import cycle
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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

def iterate_over_mask(img, mask):
    segments = np.unique(mask)
    
    for segment in segments:
        print(segment)
        yield segment, get_subimage(img, mask == segment)

def preprocess_img_hls(img_hls):
    return img_hls * (1/180, 1/255, 1/255) 

        
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

def get_image_signatures(img_list, preprocess=preprocess_img_hls, analysis_palette=np_itten_color_wheel):
    signatures = []
    for img_path in tqdm(img_list):
        img_rgb, img_hls = get_img(img_path)

        quantized_img = get_color_wheel_quantized_img(preprocess(img_hls), analysis_palette)
        signature = dict(zip(*np.unique(quantized_img, return_counts=True)))
        
        signatures.append(signature)
        
    return signatures

def get_color_wheel_quantized_img(fl_img_hls, color_wheel=np_itten_color_wheel, apply_sin_to_hue=True):
    fl_img_hls_n = fl_img_hls.copy()
    np_itten_color_wheel_n = np_itten_color_wheel.copy()

    if apply_sin_to_hue:
        fl_img_hls_n[:, :, 0] = np.sin(2*np.pi*fl_img_hls_n[:, :, 0])
        np_itten_color_wheel_n[:, 0] = np.sin(2*np.pi*np_itten_color_wheel_n[:, 0])
    
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


def quantize_img(palette_index_matrix, palette=np_itten_color_wheel):
    shape = palette_index_matrix.shape
    
    return np.take(palette, palette_index_matrix.reshape(-1), axis=0).reshape(shape[0], shape[1], 3)

def quantize_img_color_wheel(img_hls, palette=np_itten_color_wheel):
    shape = img_hls.shape
    palette_index_matrix = get_color_wheel_quantized_img(img_hls, palette)
    
    return np.take(palette, palette_index_matrix.reshape(-1), axis=0).reshape(shape[0], shape[1], 3)

def get_palette(hls_palette):
    paleta = list(map(lambda cl: colorsys.hls_to_rgb(cl[0], cl[1], cl[2]), hls_palette))
    
    return paleta

def get_img(img_path):
    img = cv2.imread(img_path)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    
    return img_rgb, img_hls

def get_luminance_weighted_mask(img_hls, mask_function=get_rule_of_thirds_mask, mask_kwargs={}):
    mask = mask_function(img_hls, **mask_kwargs)

    replacements = {}
    for segment_idx, mask_subimg in iterate_over_mask(img_hls, mask):
        replacements[segment_idx] = np.median(mask_subimg[:,:,1])
        
    flat_mask = mask.flatten()
    return pd.Series(flat_mask).map(replacements).values.reshape(*mask.shape)

def get_image_signature(img_path, preprocess=preprocess_img_hls, analysis_palette=np_itten_color_wheel, apply_sin_to_hue=True):
    img_rgb, img_hls = get_img(img_path)

    quantized_img = get_color_wheel_quantized_img(preprocess(img_hls), color_wheel=analysis_palette, apply_sin_to_hue=apply_sin_to_hue)
    signature = dict(zip(*np.unique(quantized_img, return_counts=True)))

    return signature

def get_image_signatures(img_list, preprocess=preprocess_img_hls, analysis_palette=np_itten_color_wheel, apply_sin_to_hue=True):
    signatures = []
    
    results = joblib.Parallel(n_jobs=12)(joblib.delayed(get_image_signature)(img_path, analysis_palette=analysis_palette, apply_sin_to_hue=apply_sin_to_hue) for img_path in tqdm(img_list))
            
    return results

def quantize_img_twice(img_hls, preprocess=preprocess_img_hls, palette=None, inverse_map={}):
    shape = img_hls.shape
    palette_index_matrix = get_color_wheel_quantized_img(preprocess(img_hls), palette, apply_sin_to_hue=True)
    
    return np.array(list(map(lambda x: inverse_map[x], palette_index_matrix.flatten()))).reshape(shape[0], shape[1], 3)

def get_rgb_quantize_img_twice(img_hls, preprocess=preprocess_img_hls, palette=None, inverse_map={}):
    quantized_img = quantize_img_twice(img_hls, preprocess, palette, inverse_map) * (180, 255, 255)
    
    plt.imshow(cv2.cvtColor(np.uint8(quantized_img), cv2.COLOR_HLS2RGB))
    plt.show()
    
def get_hue_palette_cube(hues):
    example_hues = [(hue, 0.5, 0.5) for hue in hues]
    
    luminances = np.array([0, 0.33, 0.66, 1])
    hue_luminance = np.transpose([np.tile(hues, len(luminances)), np.repeat(luminances, len(hues))]).copy()
    saturations = np.array([0, 0.33, 0.66, 1])

    hls_stack = []
    for saturation in saturations:
        hls_array_proto = np.zeros((hue_luminance.shape[0], hue_luminance.shape[1] + 1))
        hls_array_proto[:, :2] = hue_luminance.copy()

        hls_array_proto[:, 2] = saturation
        hls_stack.append(hls_array_proto)

    cube = np.stack(hls_stack).reshape(-1,3)
    
    pd_cube = pd.DataFrame(cube, columns=['hue', 'luminance', 'saturation'])
    pd_cube = pd_cube.sort_values(by=['hue','luminance','saturation']).reset_index(drop=True)
    pd_cube = pd_cube.reset_index()

    final_cube = pd_cube[['hue','luminance','saturation']].values.reshape(len(hues), len(luminances), len(saturations), 3)
    cube_palette = final_cube.reshape(-1, 3)
    cube_indices = np.array(list(range(len(cube_palette))))
    
    pd_cube['blanco'] = 1*(pd_cube['luminance'] == 1)
    pd_cube['negro'] = 1*(pd_cube['luminance'] == 0)
    pd_cube['gris'] = 1*(pd_cube['saturation'] == 0)*(1-pd_cube['blanco'])*(1-pd_cube['negro'])
    
    for idx, hue in enumerate(hues):
        pd_cube['hue_%s' % idx] = 1*(pd_cube['hue'] == hue)*(1-pd_cube['blanco'])*(1-pd_cube['negro'])*(1-pd_cube['gris'])
    
    blancos = list(pd_cube[pd_cube['blanco'] == 1]['index'].values)
    negros = list(pd_cube[pd_cube['negro'] == 1]['index'].values)
    gris = list(pd_cube[pd_cube['gris'] == 1]['index'].values)
    
    final_analysis_array = [
        ('negros', negros, (0, 0, 0))
        ,('grises', gris, (0, 0.5, 0))
        ,('blancos', blancos, (0, 0.85, 0))
    ] + [
        (
            'hue_%s' % hue_idx,
            list(pd_cube[pd_cube['hue_%s' % hue_idx] == 1]['index'].values), 
            example_hues[hue_idx]
        )
        for hue_idx in range(len(hues))
    ]

    final_analysis_names = [i[0] for i in final_analysis_array]
    final_analysis_palette = np.array([i[2] for i in final_analysis_array])

    inverse_map = {}
    for name, codes, color in final_analysis_array:
        for code in codes:
            inverse_map[code] = color
            
    return cube_palette, pd_cube, final_analysis_array, final_analysis_palette, final_analysis_names, inverse_map