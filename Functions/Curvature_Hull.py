"""
Author: Luis Angel Ponce Pacheco
====================

Code that find Chicken leg pose estimation base on curvature

"""
# Importing some libraries------------------------------------- 
from Functions.Curvature_Maximization import *
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Defining some functions-------------------------------------------
def curvature4cv2(curvature):
    curvature = curvature.reshape(curvature.shape[0], 1, curvature.shape[1])        
    curvature = curvature.astype(np.int32)    
    return [curvature]    


def getConvexHull(curvature):
    return cv2.convexHull(curvature[0], returnPoints=False)    
    
    
def getHullPoints(curvature):
    curvature = curvature4cv2(curvature)
    hull = []
    for i in range(len(curvature)):
        hull.append(cv2.convexHull(curvature[i])) 
    return hull[0].reshape((hull[0].shape[0],2))   
    

def getLineArray(curvature_points, max_curv_point, nx_ny):
    xc, yc = curvature_points[:,0], curvature_points[:,1]
    linex = np.linspace(xc[max_curv_point[0]], nx_ny[0][max_curv_point[0]])
    liney = np.linspace(yc[max_curv_point[0]], nx_ny[1][max_curv_point[0]])
    line = np.stack((linex, liney), axis=-1)
    return line.reshape(line.shape[0], line.shape[-1])

    
def getLinePointsInContour(line, curvature):
    points_inside = []
    for j in range(len(line)):        
        if cv2.pointPolygonTest(curvature[0], line[j], False) == 1:
            points_inside.append(line[j])
    return points_inside


# Defining main functions-------------------------------------------
def plotHullContour(curvature_points, hull_points, max_curv_point, tangents, nx_ny, index2, index_points, s, e):
    xc, yc = curvature_points[:,0], curvature_points[:,1]
    xh, yh = hull_points[:,0], hull_points[:,1]
    tx, ty = tangents[:,0], tangents[:,1]
    plt.plot(xc, yc, label='EFD Silhouette')
#    plt.plot(xh, yh, label='Convex Hull')
    plt.plot(xc[max_curv_point], yc[max_curv_point], 's', color='orange', label='Grasping Points')
    plt.plot([xc[max_curv_point[0]], tx[max_curv_point[0]]], [yc[max_curv_point[0]], ty[max_curv_point[0]]], '--', color='brown', label='Tangent')
    plt.plot([xc[max_curv_point[0]], nx_ny[0][max_curv_point[0]]], [yc[max_curv_point[0]], nx_ny[1][max_curv_point[0]]], '--', color='green', label='Orthogonal')
    plt.plot(xc[index2[1]], yc[index2[1]], 's', color='orange')

    starts_in_curv = []
    ends_in_curv = []
    index_in_curv = []
    for i in range(len(index_points)):
        start_ = curvature_points[s[i]]
        end_ = curvature_points[e[i]]
        index_ = curvature_points[index_points[i]]
        starts_in_curv.append(start_)
        ends_in_curv.append(end_)
        index_in_curv.append(index_)
        
    starts_in_curv = np.array(starts_in_curv)
    ends_in_curv = np.array(ends_in_curv)
    middle_in_curv = (starts_in_curv + ends_in_curv) / 2
    index_in_curv = np.array(index_in_curv)
    
    starts_in_curv = np.reshape(starts_in_curv, (starts_in_curv.shape[0], starts_in_curv.shape[-1]))
    ends_in_curv = np.reshape(ends_in_curv, (ends_in_curv.shape[0], ends_in_curv.shape[-1]))
    middle_in_curv = np.reshape(middle_in_curv, (middle_in_curv.shape[0], middle_in_curv.shape[-1]))
    index_in_curv = np.reshape(index_in_curv, (index_in_curv.shape[0], index_in_curv.shape[-1]))
    
    plt.plot([middle_in_curv[0][0], index_in_curv[0][0]], [middle_in_curv[0][1], index_in_curv[0][1]], color='k', label='Depth')
    plt.plot([starts_in_curv[0][0], ends_in_curv[0][0]], [starts_in_curv[0][1], ends_in_curv[0][1]],color='darkorange', label='Convex Hull')
    
    for j in range(len(index_points)):
        plt.plot([middle_in_curv[j][0], index_in_curv[j][0]], [middle_in_curv[j][1], index_in_curv[j][1]], color='k')
        plt.plot([starts_in_curv[j][0], ends_in_curv[j][0]], [starts_in_curv[j][1], ends_in_curv[j][1]],color='darkorange')

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.legend(prop={'size': 14})
    return plt


def getMaxCurvPoint(curvature):
    curvature = curvature4cv2(curvature)
    hull = getConvexHull(curvature)
    defects =  cv2.convexityDefects(curvature[0], hull)
    defects = defects.reshape(defects.shape[0], 4)  
    s, e, index_points, depth = np.hsplit(defects, 4) 
    max_depth = np.amax(depth)
    max_curv_point_id = np.where(depth == max_depth)
    max_curv_point = index_points[max_curv_point_id[0]] 
    return max_curv_point, index_points, s, e
    

def getGraspPointsHull(max_curv_point, nx_ny, curvature, curvature_points):
    curvature = curvature4cv2(curvature)
    line = getLineArray(curvature_points, max_curv_point, nx_ny)
    points_inside = getLinePointsInContour(line, curvature)
    point2 = np.array([points_inside[-1]])
    index2 = directed_hausdorff(point2, curvature_points)[1:]
    grasp_points = np.array([int(max_curv_point[0]), index2[1]])
    grasp_points = grasp_points.reshape(grasp_points.shape[0], 1)
    return grasp_points, index2

