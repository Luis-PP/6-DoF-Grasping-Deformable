"""
Author: Luis Angel Ponce Pacheco
====================

Function that plots Chicken leg pose estimation

""" 

## Adding some libraires------------------------------------------
import PIL
from PIL import Image
import os
import pickle
import numpy as np
import pyvista as pv

# Defining some functions-------------------------------------------
def dataColorPil(ipImage, path):
    """Function that returs the color image array with PIL"""
    return np.asarray(Image.open(os.path.join(path, ipImage) + "_color.jpg"))
    
    
def oneMask(mask):
    """"Function that return a single binary channel with from the instance mask"""
    mask = mask.permute(1,2,0)
    newMask = np.zeros_like(mask[:,:,[0]])
    for i in range(len(mask[0][0])):
        t = np.where(mask[:,:,[i]] == True, 1, 0)
        newMask = newMask + t
    newMask[newMask > 1] = 1
    return newMask.astype(np.uint8) * 255 
    
    
def dataXyz(ipImage, path):
    """Function that return the xyz data array"""
    return pickle.load(open(os.path.join(path, ipImage) + "_xyz.pkl", "rb"))
    

def xyzInstance(ipImage, path, mask):
    '''Function that returns the xyz data of an instance'''
    xyz = dataXyz(ipImage, path)
    xyzInstances = np.zeros_like(xyz)
    mask = oneMask(mask) / 255
    for i in range(len(mask[0][0])):
        x = np.where(mask[:,:,i] == 1, xyz[:,:,0], 0)
        y = np.where(mask[:,:,i] == 1, xyz[:,:,1], 0)
        z = np.where(mask[:,:,i] == 1, xyz[:,:,2], 0)
        xyzInstance = np.dstack((x, y, z))
        xyzInstances = xyzInstances + xyzInstance
    return xyzInstances    
        
    
def moveAndRotate(translation, rotation, mesh):
    '''Function that rotates and trasnlates a mesh'''
    mesh = mesh.translate(translation, inplace=True)
    mesh = mesh.rotate_x(rotation[0], translation, inplace=False)
    mesh = mesh.rotate_y(rotation[1], translation, inplace=False)
    mesh = mesh.rotate_z(rotation[2], translation, inplace=False)
    return mesh
    
            
# Defining main fuction-------------------------------------------
def showSinglePose(ipImage, path, kp3d, center3d, angles, center, clase, mask=None):
    """Function that plotes the instance cloud with the keypoints"""
    if mask is not None:
        xyzInst = xyzInstance(ipImage, path, mask)
        xyz = np.reshape(xyzInst, (921600,3))
    else:
        xyz = np.reshape(dataXyz(ipImage, path), (921600,3))  
# Getting the mesh color   
    rgb = np.reshape(dataColorPil(ipImage, path), (921600,3))
    elevation_cloud = pv.PolyData(kp3d)
    elevation = kp3d[:, -1]
    elevation_cloud["elevation"] = elevation 
    point_cloud = pv.Plotter()
    point_cloud.add_title('Top' if clase == 0  else 'Bottom', font='courier', color='w', font_size=40)
    point_cloud.add_points(xyz, opacity=1, point_size=1.0, render_points_as_spheres=False, scalars=rgb, rgb=True, smooth_shading=True)
# Putting labels to the keypoints    
    labels = ['1', '2', '3', '4', '5'] 
    point_cloud.add_point_labels((kp3d), labels, show_points=False, margin=0, italic=True, font_size=15, point_size=1, always_visible=True)
    point_cloud.add_points(kp3d, render_points_as_spheres=True, point_size=15.0, scalars=elevation)
# Getting center of frame and point to point at
    rotation = angles
    translation = kp3d[center]
# Getting the object frame arrows  
    file_path =  "Functions/Arrows" #'/path/to/the/arrows'
    arrowX = pv.read(os.path.join(file_path, 'ArrowX.obj'))
    arrowY = pv.read(os.path.join(file_path, 'ArrowY.obj'))
    arrowZ = pv.read(os.path.join(file_path, 'ArrowZ.obj')) 
# Moving the object grame arrows    
    arrowX = moveAndRotate(translation, rotation, arrowX)
    arrowY = moveAndRotate(translation, rotation, arrowY)
    arrowZ = moveAndRotate(translation, rotation, arrowZ)
# Adding the arrow to the main mesh   
    point_cloud.add_mesh(arrowX, color='r')
    point_cloud.add_mesh(arrowY, color='g')
    point_cloud.add_mesh(arrowZ, color='b') 
# Showing the mesh    
    light = pv.Light(position=(0, 1, 0), light_type='scene light')
    point_cloud.add_light(light)
    point_cloud.set_background('black')  
    point_cloud.show_axes()
#    point_cloud.show()
    return point_cloud


def showMultiplePose(ipImage, path, kp3d, center3d, matchT_R, clase, mask=None):
    if mask is not None:
        xyzInst = xyzInstance(ipImage, path, mask)
        xyz = np.reshape(xyzInst, (921600,3))
    else:
        xyz = np.reshape(dataXyz(ipImage, path), (921600,3))    
# Getting the mesh color   
    rgb = np.reshape(dataColorPil(ipImage, path), (921600,3))
    elevation_cloud = pv.PolyData(kp3d)
    elevation = kp3d[:, -1]
    elevation_cloud["elevation"] = elevation 
    point_cloud = pv.Plotter()
    point_cloud.add_title('Top' if clase == 0  else 'Bottom', font='courier', color='w', font_size=40)
    point_cloud.add_points(xyz, opacity=1, point_size=1.0, render_points_as_spheres=False, scalars=rgb, rgb=True, smooth_shading=True)
# Putting labels to the keypoints    
    labels = ['1', '2', '3', '4', '5'] 
    point_cloud.add_point_labels((kp3d), labels, show_points=False, margin=0, italic=True, font_size=15, point_size=1, always_visible=True)
    point_cloud.add_points(kp3d, render_points_as_spheres=True, point_size=15.0, scalars=elevation)

# adding all the candidates to the mesh    
    for i in range(len(matchT_R)):
# Getting the frame arrows mesh  
        file_path = "Functions/Arrows" #'/path/to/the/arrows'
        arrowX = pv.read(os.path.join(file_path, 'ArrowX.obj'))
        arrowY = pv.read(os.path.join(file_path, 'ArrowY.obj'))
        arrowZ = pv.read(os.path.join(file_path, 'ArrowZ.obj'))
# Getting center of frame and point to point at
        rotation = matchT_R[i][3:6]
#        print(rotation)
        translation = matchT_R[i][0:3]
#        print(translation)     
# Moving the object grame arrows    
        arrowX = moveAndRotate(translation, rotation, arrowX)
        arrowY = moveAndRotate(translation, rotation, arrowY)
        arrowZ = moveAndRotate(translation, rotation, arrowZ)
# Adding the arrow to the main mesh   
        point_cloud.add_mesh(arrowX, color='r')
        point_cloud.add_mesh(arrowY, color='g')
        point_cloud.add_mesh(arrowZ, color='b') 
# Showing the mesh    
    light = pv.Light(position=(0, 1, 0), light_type='scene light')
    point_cloud.add_light(light)
    point_cloud.set_background('black')  
    point_cloud.show_axes()
#    point_cloud.show()
    return point_cloud
    

    
def showSinglePoseGraspPoints(ipImage, path_color, path_xyz, kp3d, angles, grasping_points_3d, side, mask=None):
    """Function that plots the instance cloud with the keypoints"""
    if mask is not None:
        xyzInst = xyzInstance(ipImage, path_xyz, mask)
        xyz = np.reshape(xyzInst, ((int(xyzInst.shape[0]) * int(xyzInst.shape[1])),3))
    else:
        xyz = np.reshape(dataXyz(ipImage, path_xyz), ((int(xyzInst.shape[0]) * int(xyzInst.shape[1])),3))   
# Getting the mesh color   
    rgb = np.reshape(dataColorPil(ipImage, path_color), (int(xyz.shape[0]),3))
    elevation_cloud = pv.PolyData(kp3d)
    elevation = kp3d[:, -1]
    elevation_cloud["elevation"] = elevation 
    point_cloud = pv.Plotter()
    point_cloud.add_title(side , font='courier', color='black', font_size=40)
    point_cloud.add_points(xyz, opacity=1, point_size=1.0, render_points_as_spheres=False, scalars=rgb, rgb=True, smooth_shading=True)
# Putting labels to the keypoints    
    labels = ['1', '2', '3', '4', '5'] 
    point_cloud.add_point_labels((kp3d), labels, show_points=False, margin=0, italic=True, font_size=15, point_size=1, always_visible=True)
    point_cloud.add_points(kp3d, render_points_as_spheres=True, point_size=15.0, color='yellow')#scalars=elevation)
# Plotting the grasping points
    grasp_labels = ['G1', 'G2', ' '] 
    point_cloud.add_point_labels((grasping_points_3d), grasp_labels, show_points=False, margin=0, italic=True, font_size=15, point_size=1, always_visible=True)
    point_cloud.add_points(grasping_points_3d, render_points_as_spheres=True, point_size=15.0, color='red')    
# Getting center of frame and point to point at
    rotation = angles
    translation = grasping_points_3d[2]
# Getting the object frame arrows  
    file_path = "Functions/Arrows" #'/path/to/the/arrows'
    arrowX = pv.read(os.path.join(file_path, 'ArrowX.obj'))
    arrowY = pv.read(os.path.join(file_path, 'ArrowY.obj'))
    arrowZ = pv.read(os.path.join(file_path, 'ArrowZ.obj')) 
# Moving the object grame arrows    
    arrowX = moveAndRotate(translation, rotation, arrowX)
    arrowY = moveAndRotate(translation, rotation, arrowY)
    arrowZ = moveAndRotate(translation, rotation, arrowZ)
# Adding the arrow to the main mesh   
    point_cloud.add_mesh(arrowX, color='r')
    point_cloud.add_mesh(arrowY, color='g')
    point_cloud.add_mesh(arrowZ, color='b') 
# Showing the mesh    
    light = pv.Light(position=(0, 1, 0), light_type='scene light')
    point_cloud.add_light(light)
    point_cloud.show_grid(color='black')
    point_cloud.set_background('#FAF5EF')
#    point_cloud.show_grid(color='w')
#    point_cloud.set_background('black')
    point_cloud.show_axes()
#    point_cloud.show()
    return point_cloud       
    
