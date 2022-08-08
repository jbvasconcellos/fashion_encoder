import json
import pandas as pd
import numpy as np

from PIL import Image, ImageDraw, ImageOps
import os

import torch
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
# import cv2
import matplotlib.pyplot as plt

from glob import glob

from datetime import datetime


def box_selection_measures(list_, min_width=35, min_height=35, use_prop=True, prop=2000):
    """Filters bounding boxes that don't match minimun size and proportion (if prot=True)"""
        
    if len(list_) != 4:
        raise ValueError("List length does not match")
    
    else:
        x0 = list_[0]
        y0 = list_[1]
        x1 = list_[2]
        y1 = list_[3]

        w = abs(x1 - x0)
        h = abs(y1 - y0)
        
        if use_prop:
            if (w >= min_width) & (h >= min_height) & (w*h >= prop):
                return list_
            else: return np.nan
        else:
            if (w >= min_width) & (h >= min_height):
                return list_
            else: return np.nan

def tuple_inv_in_list(list_): # used to correct order of arguments in segmentatations list in DeepFashion2
    """Inverts tuples in odd positions in list"""
    for i in range(len(list_)):
        if i % 2 == 0:
            pass
        else:
            list_[i] = list_[i][::-1] # inverts tuple
    return list_

def plot_bounding_box(image, annotations_file): # for visualization only
    """
        Plots sliced bboxxes with white background from input image
    """
    
    # extract boxxes, categories, polygons for each item in annot
    boxxes, categories, polygons = {}, {}, {}
    for i in range(2): #removes 'source and 'pair_id and keeps just item1, item2;
        i += 1
        boxxes[i] = np.array(annotations_file['item'+str(i)]['bounding_box']) 
        categories[i] = annotations_file['item'+str(i)]['category_name']
        #category2 = annotations_file['item2']['category_name']
        polygons[i] = annotations_file['item'+str(i)]['segmentation']

    plotted_image = ImageDraw.Draw(image)
    
    fig = plt.figure(figsize=(10, 30))
    
    # separate items
    new_images = []
    for box_id, box in boxxes.items():
        
        transformed_annotations = np.copy(box)
        x0 = transformed_annotations[0]
        y0 = transformed_annotations[3]
        x1 = transformed_annotations[2]
        y1 = transformed_annotations[1]
        
        # for example visualization: plot separate item 
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        polygon = [(polygons[box_id][0][i], polygons[box_id][0][i+1]) for i in range(len(polygons[box_id][0]) -1)]
        polygon = tuple_inv_in_list(polygon)
        plotted_image.text((x0, y0 - 10), categories[box_id]) 
        plotted_image.polygon(polygon, outline ="blue")
    
        # for embedding: get individual item without background
        bg_img = Image.new("RGB", image.size, (255, 255, 255) )
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon, fill=255)
        
        new_image = Image.composite(image, bg_img, mask).crop(box)
        new_images.append(new_image)
        
    return new_images

def get_bboxxes(list_):
    """Extracts bboxxes from annot files.
        list_ : list of annotatio files
        output: dict {img_id : [[bbox1, bbox2], [cat_bbox1, cat_bbox2]]}
    """
    img_bbox_dict = {}
    for annotation in list_:
        bboxxes = []
        categories = []
        segmentations = []
        with open(annotation, 'r') as file:
            try:
                data = {key:value for key, value in json.load(file).items() if key in ['item1', 'item2'] } # only item1,..., itemN if key not in  ['source', 'pair_id']
                for item in data.keys():
                    bboxxes.append(list({key:value for key, value in data[item].items() if(key  == 'bounding_box')}.values())[0] )
                    categories.append(list({key:value for key, value in data[item].items() if(key  == 'category_id')}.values())[0] )
                    segmentations.append(list({key:value for key, value in data[item].items() if(key  == 'segmentation')}.values())[0] )
            except: import pdb; pdb.set_trace()
        img_bbox_dict[annotation[-11:-5]] = [bboxxes, categories, segmentations] # key=index; value=bbox
#         img_categs_dict[annotation[-11:-5]] = categories
        
    return img_bbox_dict

def bbox_croper(image, box, dataset_name : str): # used in dataset class
    """
        Produces sliced bboxxes from input image
            image: PIL.Image object
            box: list [x0, y0, x1, y1]
        Output: PIL image
    """
    
    # extract boxxes, categories, polygons for each item in annot
    plotted_image = ImageDraw.Draw(image)
    

    transformed_annotations = np.copy(box)
    x0 = transformed_annotations[0]
    y0 = transformed_annotations[1] # 3
    x1 = transformed_annotations[2]
    y1 = transformed_annotations[3] #1
    
    if dataset_name == 'DeepFashion2':
        crop_box = [x0,y0,x1,y1]
    elif dataset_name == 'Fashionpedia':
        crop_box = [x0, y0, x0+x1, y0+y1]
    else: raise ValueError("Uknown dataset")
        
    return image.crop(crop_box) #(box) 

# def look_for_smaller_images(df_):
#     """
#         Returns list of bad images indexes
#     """
#     bad_images =[]
#     for idx in range(len(df_)):
#         read = plt.imread(df_.img_path[idx])
#         h, w, c = read.shape
#         if (h < 256) or (w < 128) or (c!=3):
#             bad_images.append(idx)

#     return bad_images

def look_for_smaller_images(df_, base_w, base_h):
    """
        Returns list of bad images indexes
    """
    bad_images =[]
    for idx in range(len(df_)):
        read = Image.open(df_.img_path[idx]) #plt.imread(df_.img_path[idx])
        w, h = read.size
        if (h < base_h) or (w < base_w):# or (read.mode != 'RGB'):
            bad_images.append(idx)

    return bad_images

def get_idxs(list_, annotations=True):
    '''Gets images and anotations indexes'''
    if annotations:
        annot_indices = [string[-11:-5] for string in list_]
        return annot_indices
    else:
        imgs_indices = [string[-10:-4] for string in list_]
        return imgs_indices
    
def df_constructor(list_, img_path):
    """
        Produces df from list of images and annotations
            list_ : annot files
            img_path: list of image files
    """
    bboxxes = get_bboxxes(list_)
    df = pd.DataFrame.from_dict(bboxxes, orient='index', columns=['bounding_boxxes', 'category_id', 'segmentation']) # 'img_id'
    df = df.apply(pd.Series.explode)
    df = df.reset_index().rename(columns={'index':'img_id'})
    df['img_path'] = df['img_id'].apply(lambda x: img_path+x+'.jpg')
#     df = df.explode('segmentation')
    df = df[df['img_path'].apply(lambda x : check_image_file(x) )]
#     df = df.reset_index(drop=True)
    return df

def check_image_file(img_path_):
    return os.path.isfile(img_path_)

def bbox_BG_remover(image, box, polygon): # Emb 5
    """
        Produces sliced bboxxes with white background from input image
    """

    plotted_image = ImageDraw.Draw(image)
    
    transformed_annotations = np.copy(box)
    x0 = transformed_annotations[0]
    y0 = transformed_annotations[1] # 3
    x1 = transformed_annotations[2]
    y1 = transformed_annotations[3] #1

    # for example visualization: plot separate item 
    plotted_image.rectangle(((x0,y0), (x1,y1)))
    polygon = [(polygon[i], polygon[i+1]) for i in range(len(polygon) -1)]
    polygon = tuple_inv_in_list(polygon)
#         plotted_image.text((x0, y0 - 10), categories[box_id]) 
    plotted_image.polygon(polygon, outline ="blue")

    # for embedding: get individual item without background
    bg_img = Image.new("RGB", image.size, (255, 255, 255) )
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)

    new_image = Image.composite(image, bg_img, mask).crop(box)
    #new_images.append(new_image)
        
    return new_image
# ============================================================================================================================= #
# Debuggers

from tqdm import tqdm


def check_PILImage(img_):
    PIL_img  = Image.open(img_).convert('RGB')
    return type(PIL_img) == Image.Image