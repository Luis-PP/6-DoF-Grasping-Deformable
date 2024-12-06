"""
Author: Luis Angel Ponce Pacheco
====================

Code that find Chicken leg pose estimation using the Curvature Maximization approach

"""

# Importing some libraries------------------------------------- 
import cv2
import os
import pickle
import numpy as np
import math as m
from scipy.spatial.transform import Rotation as R
# Calling main functions
from Functions.FindAngles import (getRotationMatrix, getAllAngles, getRotatMatrixFrom2Points, getSide)
from Functions.PlotGPE import (showSinglePose, showMultiplePose, showSinglePoseGraspPoints)
from Functions.Curvature_Maximization import(getMaxConcaveCurvature, label2points, plotfrictionCone, getFCpoints, plotFC, getGraspingPoints, getWidth)
from Functions.FindTargetObject import (getTargetObject)

# Import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# Define image and paths-------------------------------------------
imageId = str(48736) #"image_id" 
path =  "data" #"/path/to/data"
path_color =  "data/color" #"/path/to/data"
path_xyz =  "data/xyz" #'/path/to/the/xyz/data'
yamlFile =  "Configs/keypoint_rcnn_R_50_FPN_3x.yaml" #"/path/to/yaml/file.yaml"
weightsFile =  "model_final.pth" #"/path/to/weghts/file/model_final.pth"
#n = 5 # intance of the picture
c = 2 # key point to be consider as center
mu = 0.4 # friction coneficient

# Defining some functions-------------------------------------------
def dataColor(ipImage, path):
    """Function that return the color image array"""
    return cv2.imread(os.path.join(path, ipImage) + "_color.jpg")


def outputsD2(image, yamlFile, weightsFile, device="cpu", threshold=0.75, instance=None, scale=1):
    """Function that returns the outputs of the trained Detectron2 model of a new image, 
and draw the model outputs on an image."""
    cfg = get_cfg()
    cfg.MODEL.DEVICE = device
    cfg.merge_from_file(yamlFile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold 
    cfg.MODEL.WEIGHTS = (weightsFile)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
    scale=scale)
    if instance is not None:
        out = v.draw_instance_predictions(outputs["instances"]
        [instance].to("cpu"))
    else:
        out = v.draw_instance_predictions(outputs["instances"][:].to("cpu"))    
    return outputs, out.get_image()[:, :, ::-1]
    
    
def showCv2(image):
    '''Function that shows and image with cv2'''
    cv2.imshow(str(image), image)
    cv2.waitKey(0)  
  
      
def getKpCoordin(outputs, instance=None):
    """Function that returns the predicted keypoints cordinates of the model 
instace allow to select one specific or all (defalut). The output is 
(N, num_keypoint, 2). Each row in the last dimension is (x, y)"""
    if instance is not None:
        return np.delete(outputs["instances"][instance].pred_keypoints.numpy(), -1, axis=2)
    else:
        return np.delete(outputs["instances"][:].pred_keypoints.numpy(), -1, axis=2)
        
  
def columnMove(array, order=[1,0]):
    """Functiion that moves a column of an array given the oerder. order is an array 
(e.i. [1,0]) moves the second to the first (defaul)"""
    return array[:,order]
    
def dataXyz(ipImage, path):
    """Function that return the xyz array"""
    return pickle.load(open(os.path.join(path, ipImage) + "_xyz.pkl", "rb"))
          
def getKp3d(ipImage, path, kpCoordin, instance=None):
    """Function that return the xyz values from a xyz file, given the keypoints coodinates""" 
    xyzAll = dataXyz(ipImage, path)
    if instance is not None:
        singleKpCoord = kpCoordin [instance,:,:]
        singleKpCoordMoved = columnMove(singleKpCoord)
        xs = ys = zs = [0.0]
        for j in range(len(singleKpCoordMoved)):
            x = [(xyzAll[int(singleKpCoordMoved[j,0]), 
            int(singleKpCoordMoved[j,1]), 0])]
            xs = np.append(xs, x, axis = 0)
            y = [(xyzAll[int(singleKpCoordMoved[j,0]), 
            int(singleKpCoordMoved[j,1]), 1])]
            ys = np.append(ys, y, axis = 0)
            z = [(xyzAll[int(singleKpCoordMoved[j,0]), 
            int(singleKpCoordMoved[j,1]), 2])]
            zs = np.append(zs, z, axis = 0)
        xyzPoints = np.hstack(((np.array([xs])).T, 
        (np.array([ys])).T, (np.array([zs])).T))
        return np.delete(xyzPoints, 0, 0)    
    else:
        allNewXyzPoints = np.zeros([1, 3])
        for i in range(len(kpCoordin)):
            singleKpCoord = kpCoordin [i,:,:]
            singleKpCoordMoved = columnMove(singleKpCoord)
            xs = ys = zs = [0.0]
            for j in range(len(singleKpCoordMoved)):
                x = [(xyzAll[int(singleKpCoordMoved[j,0]), 
                int(singleKpCoordMoved[j,1]), 0])]
                xs = np.append(xs, x, axis = 0)
                y = [(xyzAll[int(singleKpCoordMoved[j,0]), 
                int(singleKpCoordMoved[j,1]), 1])]
                ys = np.append(ys, y, axis = 0)
                z = [(xyzAll[int(singleKpCoordMoved[j,0]), 
                int(singleKpCoordMoved[j,1]), 2])]
                zs = np.append(zs, z, axis = 0)
            xyzPoints = np.hstack(((np.array([xs])).T, 
            (np.array([ys])).T, (np.array([zs])).T))
            newXyzPoints = np.delete(xyzPoints, 0, 0)
            allNewXyzPoints = np.concatenate((allNewXyzPoints, newXyzPoints))
        return np.delete(allNewXyzPoints, 0, 0)  
        

        
        
def binaryInstance(mask):
    """Function that returns a balck and white image to visualize the isntances"""
    binary = mask.permute(1,2,0).long().numpy().astype(np.uint8) * 255
    binary[binary > 255] = 255
    return binary

    
    
def getClase(outputs, instance=None):
    """Function that returns the class of the instances from the trained model
    Instace allow to select one specific or all (defalut)."""
    if instance is not None:
        return outputs["instances"][instance].pred_classes
    else:
        return outputs["instances"][:].pred_classes
        
        
def getScores(outputs, instance=None):
    """Function that returns the class of the instances from the trained model
    Instace allow to select one specific or all (defalut)."""
    if instance is not None:
        return outputs["instances"][instance].scores
    else:
        return outputs["instances"][:].scores

def getNeighborhood(kpCoordin, radio):
    piece_list = []        
    for piece in kpCoordin:
        kp_list = []
        for kp in piece:
            neighbor_list = []
            for x in range(-radio, radio+1):
                for y in range(-radio, radio+1):
                    neighbor = [kp[0] + x, kp[1] + y]
                    neighbor_list.append(neighbor)
            kp_list.append(neighbor_list)        
        piece_list.append(kp_list)   
    kp_neighborhood = np.array(piece_list)
    return np.reshape(kp_neighborhood, (kp_neighborhood.shape[0], (kp_neighborhood.shape[1] * kp_neighborhood.shape[2]), kp_neighborhood.shape[3]))
    
def getMeanNehigbor(kp3d_neighbor_3d, kpCoordin):
    kp3d_neighbor_3d = np.reshape(kp3d_neighbor_3d, (kpCoordin.shape[0], kpCoordin.shape[1], int((kp3d_neighbor_3d.shape[0] / (kpCoordin.shape[0]*kpCoordin.shape[1]))), kp3d_neighbor_3d.shape[-1]))
    return np.nanmean(kp3d_neighbor_3d, axis=2)                
                               
# Run the code-----------------------------------------------------
if __name__ == "__main__":
    rgb = dataColor(imageId, path_color)
    showCv2(rgb)
    outputs, imOutputs = outputsD2(rgb, yamlFile, weightsFile)
#    print(getClase(outputs), getScores(outputs))
    showCv2(imOutputs)
    kpCoordin = getKpCoordin(outputs)
    
    kp_neighborhood = getNeighborhood(kpCoordin, radio=5)
    kp3d_neighbor_3d = getKp3d(imageId, path_xyz, kp_neighborhood)    
    kp3d_neighbor_3d_mean = getMeanNehigbor(kp3d_neighbor_3d, kpCoordin)
    kp3d_all = kp3d_neighbor_3d_mean
    
    kp3d_all = getKp3d(imageId, path_xyz, kpCoordin)
    kp3d_all = kp3d_all.reshape(int(kp3d_all.shape[0]/5), 5, 3)
    
    n, kp3d, mask, clase = getTargetObject(outputs, imageId, path_xyz, kp3d_all)
    
    if n < 0:
        print('Program interrupted, No Object found')
        exit()

    binary = binaryInstance(mask)
    showCv2(binary)
    side = getSide(kp3d, clase)
    
# Getting and plotting force closure ------------------------------
    for i in range(100):
        i += 1
        print(i)
        contours, curvature, concaves, curvature_maximos, dist, tangents = getMaxConcaveCurvature(binary, order=i, num_points=300)
        fcPoints = getFCpoints(dist, curvature, tangents, concaves, mu)
        if fcPoints[0] != None:
            print('order:', i)
            print('Force closure at: ')
            print(fcPoints, fcPoints.shape)
            break
        else:
            continue 
    toPlot = label2points(curvature, concaves, dist, curvature_maximos)
    
    toPlot = plotfrictionCone(curvature, concaves, tangents, mu, toPlot)
    
    toPlot = plotFC(fcPoints, curvature, concaves, tangents, mu, toPlot)
    grasping_points = getGraspingPoints(concaves, fcPoints, contours)
    print('Grasp at points: ')
    print(grasping_points, grasping_points.shape)
    toPlot.show() 

    
# Getting the Object Pose given the two grasping points ----------------
    grasping_points_3d = getKp3d(imageId, path_xyz, np.array([grasping_points]))
    print('grasping_points_3d: ')
    print(grasping_points_3d)
    width = getWidth(grasping_points_3d)
    print('width:', width)
    rotat_matrix_curva = getRotatMatrixFrom2Points(kp3d, grasping_points_3d, clase=clase)
    print('rotat_matrix_curva: ')
    print(rotat_matrix_curva)
    r_curva = R.from_matrix(rotat_matrix_curva)
    angles_curva = r_curva.as_euler('xyz', degrees = True)
#    angles_curva = [angles_curva[0], angles_curva[1]-90, angles_curva[2]]
    print('angles_curva:')
    print(angles_curva)    
# Showing the grasp    
    pose_curva = showSinglePoseGraspPoints(imageId, path_color, path_xyz, kp3d,  angles_curva, grasping_points_3d, side, mask=mask)
    pose_curva.show()    
       
    
