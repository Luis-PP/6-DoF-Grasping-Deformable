"""
Author: Luis Angel Ponce Pacheco
====================

Code that compares Curvature Maximization and Convex Hull methodologies to find the grasping points

"""

# Importing some libraries------------------------------------- 
import cv2
import os
from os import listdir
import time
import numpy as np
import math as m
# To plot 
import numpy as np

# Calling main functions
from Functions.Curvature_Maximization import(getMaxConcaveCurvature, label2points, plotfrictionCone, getFCpoints, plotFC, getGraspingPoints, getWidth, getContours, changeLengthVector, rotateTanget)
from Functions.Curvature_Hull import (getMaxCurvPoint, getGraspPointsHull, plotHullContour, getHullPoints)

# Import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog




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
    
def saveCv2(image, directory, file_name):
    os.chdir(directory)
    cv2.imwrite(file_name + '_grsp_pnts.jpg', image)
    
def addCenterCv2(image, ceter, instance):
    color = (0,0,0)#(10,0,10)
    image = cv2.circle(image, (center[0][0], center[0][1]), 8, color, 4)
    image = cv2.circle(image, (center[0][0], center[0][1]), 4, color, -1)
    # image = cv2.putText(image, str(instance), (center[0][0]+15, center[0][1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return image

def addLegend(image):
    image = cv2.putText(image, 'Convex Hull', (1098, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    image = cv2.circle(image, (1080, 15), 8, (0,0,255), 2)
    image = cv2.circle(image, (1080, 15), 3, (0,0,255), -1)
    image = cv2.putText(image, 'Force Closure', (1098, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
    image = cv2.circle(image, (1080, 42), 8, (255,0,0), 2)
    image = cv2.circle(image, (1080, 42), 3, (255,0,0), -1)
    image = cv2.putText(image, 'Centroid', (1098, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,0,10), 2, cv2.LINE_AA)
    image = cv2.circle(image, (1080, 69), 8, (10,0,10), 2)
    image = cv2.circle(image, (1080, 69), 3, (10,0,10), -1)
    return image

    
def addGrapingPointsCv2(image, center, color):
    if color == 'red':
        color = (0, 0, 255)
    elif color == 'blue':
        color = (255, 0, 0)
    if image.shape[-1] == 1:
        image2 = np.zeros((image.shape[0], image.shape[1], 3))
        image2[:,:,0] = image[:,:,0]
        image2[:,:,1] = image[:,:,0]
        image2[:,:,2] = image[:,:,0]
    elif image.shape[-1] > 1:
        image2 = image
    
    image2 = cv2.circle(image2, (int(center[0][0]), int(center[0][1])), 8, color, 4)
    image2 = cv2.circle(image2, (int(center[1][0]), int(center[1][1])), 8, color, 4)
    # image2 = cv2.circle(image2, (int(center[2][0]), int(center[2][1])), 5, color, 2)
    image2 = cv2.circle(image2, (int(center[0][0]), int(center[0][1])), 4, color, -1)
    image2 = cv2.circle(image2, (int(center[1][0]), int(center[1][1])), 4, color, -1)
    # image2 = cv2.circle(image2, (int(center[2][0]), int(center[2][1])), 2, color, -1)
    return image2
        


def getInstance(outputs, instance=None):
    """Function that returns the the mask of the instances from the trained model
Instace allow to select one specific or all (defalut). The mask is bolean"""
    if instance is not None:
        return outputs["instances"][instance].pred_masks
    else:
        return outputs["instances"][:].pred_masks
        
        
def binaryInstance(mask):
    """Function that returns a balck and white image to visualize the isntances"""
    binary = mask.permute(1,2,0).long().numpy().astype(np.uint8) * 255
    binary[binary > 255] = 255
    return binary


def get2dCenter(image):
    """Function that returns centroid of a given image, it must be shape (x,y,n) with
n = number of instances"""
    centroids =[]
    for i in range(len(image[0][0])): 
        ret,thresh = cv2.threshold(image[:,:,i],127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)
        if len(contours) > 1:
            sizes = []
            for i in range(len(contours)):
                size = len(contours[i])
                sizes.append(size)
            max_size = max(sizes)
            index = sizes.index(max_size)
            contours = [contours[index]] 
        for j in contours:
            M = cv2.moments(j)
            if M["m00"] == 0.0:
                value = 1e-3
            else:
                value = M["m00"]
            cX = int(float(M["m10"]) / float(value))
            cY = int(float(M["m01"]) / float(value))
            center = [cX, cY]
            centroids.append(center)
    return centroids    
    
    
def getClase(outputs, instance=None):
    """Function that returns the class of the instances from the trained model
    Instace allow to select one specific or all (defalut)."""
    if instance is not None:
        return outputs["instances"][instance].pred_classes
    else:
        return outputs["instances"][:].pred_classes
        
    
                               
# Run the code-----------------------------------------------------
if __name__ == "__main__":
# Define image and paths-------------------------------------------
    CH_folder = []
    FC_folder = []
    path = "data/color"
    save_directory = '/path/to/save/the/new/data'
    grasp_width = 140
    mu = 0.4  # friction coefficient Find it at # https://www.utwente.nl/en/et/ms3/research-chairs/rj-temp/stt-archive/research/publications/phd-theses/thesis_veijgen.pdf
    for picture in os.listdir("data/color"):
        CH_image = []
        FC_image = []
        imageId = picture[:-10]
        print('imageId', str(imageId))
        path = "data/color"
        yamlFile = "Configs/keypoint_rcnn_R_50_FPN_3x.yaml"
        weightsFile = "model_final.pth"

        start_time = time.time()
        rgb = dataColor(imageId, path)
        outputs, imOutputs = outputsD2(rgb, yamlFile, weightsFile)
        showCv2(imOutputs)


        clase = getClase(outputs)   
        binarys = np.zeros_like(rgb)
        j = 0
        for i in range(len(clase)):
            if clase[i] == 1 or clase[i] == 3:
                j = j + 1
                n = i
                mask = getInstance(outputs, instance=n)
                binary = binaryInstance(mask)
                center = get2dCenter(binary)
#                showCv2(binary)

    ########################################################################
    # Force Closure ------------------------------
                for k in range(100):
                    k += 1
                    contours, curvature, concaves, curvature_maximos, dist, tangents = getMaxConcaveCurvature(binary, order=k, num_points=300)
                    fcPoints = getFCpoints(dist, curvature, tangents, concaves, mu)
                    if fcPoints[0] != None:
#                        print('order:', k)
#                        print('Force closure at: ')
#                        print(fcPoints, fcPoints.shape)
                        break
                    else:
                        continue 
                grasping_points_FC = getGraspingPoints(concaves, fcPoints, contours)

    ########################################################################
    # Convex Hull----------------------------------------------------
                contours, curvature, concaves, curvature_maximos, dist, tangents = getMaxConcaveCurvature(binary, order=8, num_points=300) 
                tangents = changeLengthVector(tangents, curvature, length=150) #change length if needed
                nx_ny = rotateTanget(curvature, tangents)
                hull_points = getHullPoints(curvature)
                max_curv_point = getMaxCurvPoint(curvature)
                grasp_points, index2 = getGraspPointsHull(max_curv_point, nx_ny, curvature, curvature)
                grasping_points_CH = getGraspingPoints(curvature, grasp_points, contours)
    ########################################################################

    # Plotting the grasping points in the image
                binary = addGrapingPointsCv2(binary, grasping_points_CH, color='red')
                binary = addGrapingPointsCv2(binary, grasping_points_FC, color='blue')
                binary = addCenterCv2(binary, center, j)
#                showCv2(binary)
                binarys = binarys + binary
                
                dist_CH_center = m.dist(center[0], grasping_points_CH[-1])
                dist_FC_center = m.dist(center[0], grasping_points_FC[-1])
                
                width_CH = m.dist(grasping_points_CH[0], grasping_points_CH[1])
                width_FC = m.dist(grasping_points_FC[0], grasping_points_FC[1])
                print('CH:', width_CH, '   FC:', width_FC)
                
                if width_CH > grasp_width and width_FC > grasp_width:
                    CH_image.append(0)
                    FC_image.append(0)
                elif width_CH > grasp_width and width_FC <= grasp_width:
                    CH_image.append(0)
                    FC_image.append(1)
                elif width_CH <= grasp_width and width_FC > grasp_width:
                    CH_image.append(1)
                    FC_image.append(0)
                elif width_CH <= grasp_width and width_FC <= grasp_width:
                    if dist_CH_center < dist_FC_center:
                        CH_image.append(1)
                        FC_image.append(0)
                    elif dist_CH_center > dist_FC_center:
                        CH_image.append(0)
                        FC_image.append(1)
                    
                
        print('CH_image: ', CH_image)
        print('FC_image: ', FC_image)

        
        CH_folder.append(CH_image)
        FC_folder.append(FC_image)
#        CH_folder.append(statistics.mean(CH_image))
#        FC_folder.append(statistics.mean(FC_image))
        
        binarys = addLegend(binarys)
        showCv2(binarys)

#        saveCv2(binarys, save_directory, imageId) #Uncomment to save the images
    
    print('CH_folder: ', CH_folder)
    print('FC_folder: ', FC_folder)
    

########################################################################
# Saving the results
    CH_FC = [CH_folder, FC_folder]
    print('CH_FC: ', CH_FC)

    # with open('File_name', 'w') as f:
      
        # using csv.writer method from CSV package
#        write = csv.writer(f) #Uncomment to save the outputs
#        write.writerows(CH_FC) #Uncomment to save the outputs







