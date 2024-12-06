"""
Author: Luis Angel Ponce Pacheco
====================

Code that find the target Chicken leg among the pile

"""

# Importing some libraries------------------------------------- 
import numpy as np
from scipy.spatial.distance import (pdist, squareform)
from .PlotGPE import (xyzInstance)
import matplotlib.pyplot as plt
import os
import pickle

   
# Defining some functions-------------------------------------------


def getInstance(outputs, instance=None):
    """Function that returns the the mask of the instances from the trained model
Instace allow to select one specific or all (defalut). The mask is bolean"""
    if instance is not None:
        return outputs["instances"][instance].pred_masks
    else:
        return outputs["instances"][:].pred_masks
        
         
def getClase(outputs, instance=None):
    """Function that returns the class of the instances from the trained model
    Instace allow to select one specific or all (defalut)."""
    if instance is not None:
        return outputs["instances"][instance].pred_classes
    else:
        return outputs["instances"][:].pred_classes      
                                         

def getBoxes(outputs, instance=None):
    """Function that returns the class of the instances from the trained model
    Instace allow to select one specific or all (defalut)."""
    if instance is not None:
        return outputs["instances"][instance].pred_boxes
    else:
        return outputs["instances"][:].pred_boxes

                               
def getFarthestPoint(points):
    dist = pdist(points)
    dist = squareform(dist);
#    N, [I_row, I_col] = np.nanmax(dist), np.unravel_index(np.argmax(dist), dist.shape)
    dist_means = np.mean(dist, axis=1)
    max_dist = np.argmax(dist_means)
    return int(max_dist)


def getBoxesCenter(outputs, clases, imageId, path):
    boxes = getBoxes(outputs)
    centers = boxes.get_centers()
    return centers.numpy()   


def getInstanceMeans(outputs, clases, imageId, path):
    means = np.array([])
    for i in range(len(clases)):
        mask = getInstance(outputs, i)
        xyzInst = xyzInstance(imageId, path, mask)
        xyz = np.reshape(xyzInst, ((int(xyzInst.shape[0]) * int(xyzInst.shape[1])),3))
        xy = np.delete(xyz, 2, 1)
        xy[xy == 0] = np.nan
        mean = np.nanmean(xy, axis=0)
        means = np.append(means, mean)
    return means.reshape(len(clases), 2)
    

def removeIndexNan(key_points, means):  
    nan_indexs = np.argwhere(np.isnan(key_points))
    if nan_indexs.shape[0] == 0:
        means_No_NaN = means
    else:    
        nan_indexs = np.unique(nan_indexs[:,0])
        means_No_NaN = np.delete(means, nan_indexs, axis=0)
    return means_No_NaN  

    
# Defining main function-------------------------------------------
def getTargetObject(outputs, imageId, path, kp3d_all):  
    clases = getClase(outputs)
    mask = getInstance(outputs)
    means = getInstanceMeans(outputs, clases, imageId, path)
    means_No_NaN = removeIndexNan(kp3d_all, means)	
    centers = getBoxesCenter(outputs, clases, imageId, path)

    if len(centers) == 0:
        n = -2
        return n, kp3d_all, mask, clases
    
    for i in range(len(centers)):     
        if len(centers) <= 2:
            index_class = i
        else:
        # Farthest average distance
            dist = pdist(centers)
            dist = squareform(dist)
            dist_centers = np.mean(dist, axis=1)
            sort_dist_centers = np.sort(dist_centers)
            max_dist_index = np.where(dist_centers == sort_dist_centers[-i-1])       
            index_class = int(max_dist_index[0])
        
        if np.isnan(kp3d_all[index_class]).any():       
            n = -1
            print('n: ', n)
            print('----NaN IN PIECE FOUND----')             
        else:
            if clases[index_class] == 1 or clases[index_class] == 3:
                n = index_class
                print('n: ', n)
                print('----WHOLE PIECE FOUND----') 
                break          
            else:
                n = index_class
                print('n: ', n)
                print('----NO WHOLE PIECE FOUND----')
    n = int(n)
    kp3d = kp3d_all[n]
    mask = mask[n].reshape(1, mask[n].shape[-2], mask[n].shape[-1])
    clase = clases[n] 
    return n, kp3d, mask, clase 


