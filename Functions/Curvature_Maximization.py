"""
Author: Luis Angel Ponce Pacheco
====================

Code that find Chicken leg pose estimation base on curvature

"""

# Importing some libraries------------------------------------- 
import cv2
import math as m
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyefd import (reconstruct_contour, elliptic_fourier_descriptors)
from scipy.spatial.distance import directed_hausdorff

# Defining some functions-------------------------------------------
def getTangentCurvature(coeffs, locus=(0, 0), num_points=300):
    '''Function that returns the tangent vectors of the curvature,
    first derivative'''
    t = np.linspace(0, 1.0, num_points)
    coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 1)
    orders = coeffs.shape[0]
    orders = np.arange(1, orders + 1).reshape(-1, 1)
    order_phases = 2 * orders * np.pi * t.reshape(1, -1)
    xt_all = - (coeffs[:, 0] * (2 * orders * np.pi) * np.sin(order_phases)) + (coeffs[:, 1] * (2 * orders * np.pi) * np.cos(order_phases))
    yt_all = - (coeffs[:, 2] * (2 * orders * np.pi) * np.sin(order_phases)) + (coeffs[:, 3] * (2 * orders * np.pi) * np.cos(order_phases))
    xt_all = xt_all.sum(axis=0)
    yt_all = yt_all.sum(axis=0)
    xt_all = xt_all + np.ones((num_points,)) * locus[0]
    yt_all = yt_all + np.ones((num_points,)) * locus[1]
    return np.stack([xt_all, yt_all], axis=1)
    
    
def getNormalsCurvature(coeffs, locus=(0, 0), num_points=300):
    '''Function that returns the normal vectors of the curvature
    second derivative'''
    t = np.linspace(0, 1.0, num_points)
    coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 1)
    orders = coeffs.shape[0]
    orders = np.arange(1, orders + 1).reshape(-1, 1)
    order_phases = 2 * orders * np.pi * t.reshape(1, -1)
    xt_all = - ((coeffs[:, 0] * ((order_phases)**2)) * np.cos(order_phases)) - ((coeffs[:, 1] * ((order_phases)**2)) * np.sin(order_phases))
    yt_all = - ((coeffs[:, 2] * ((order_phases)**2)) * np.cos(order_phases)) - ((coeffs[:, 3] * ((order_phases)**2)) * np.sin(order_phases))    
    xt_all = xt_all.sum(axis=0)
    yt_all = yt_all.sum(axis=0)
    xt_all = xt_all + np.ones((num_points,)) * locus[0]
    yt_all = yt_all + np.ones((num_points,)) * locus[1]
    return np.stack([xt_all, yt_all], axis=1)


def getContours(imbinary):
    '''Function that returns the contours of a binary image'''   
    contours, hierarchy = cv2.findContours(imbinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        sizes = []
        for i in range(len(contours)):
            size = len(contours[i])
            sizes.append(size)
        max_size = max(sizes)
        index = sizes.index(max_size)                
        contours = [contours[index]]        
    return contours
    
    
def getContoursMean(contours):
    contours = contours[0].reshape((contours[0].shape[0],2))
    xMean = np.mean(contours[:,0])
    yMean = np.mean(contours[:,1])
    return np.array([xMean, yMean])
    
    
def getEFDCoeff(order, contours):
    '''Funcation that returns the EFD coefficients'''
    coeffs = []
    for cnt in contours:
        # Find the coefficients of all contours
        coeffs.append(elliptic_fourier_descriptors(np.squeeze(cnt), order=order, normalize=False))
    coeffs = np.array(coeffs)
    return coeffs.reshape((order,(int(coeffs.shape[0]) * 4)))
    
    
def getEuclideanNorm(normals, curvature):
    '''Function that returns the euclidean distance between x and y of 
    a norm vector''' 
    dist = np.zeros(1)
    for i in range(len(normals)):
        vector_start = np.array([curvature[i,0], curvature[i,1]])
        vector_end = np.array([normals[i,0], normals[i,1]])
        d = np.linalg.norm(vector_start - vector_end)
        dist = np.vstack((dist, d))
    dist = np.delete(dist,0,0)   
    dist[0] = 0.0 
    return dist

    
def getMaxCurvatures(dist):
    '''Function that returns pics of an array'''
    maximos = np.r_[True, dist[:,0][1:] > dist[:,0][:-1]] & np.r_[dist[:,0][:-1] > dist[:,0][1:], True]
    curvature_maximos = np.where(maximos == True, dist[:,0], maximos)
    return maximos, curvature_maximos
                
    
def getConcaves(curvature_maximos, dist, curvature):  
    '''Function that return an array with none zero value if the curvature is
    convave, and zero otherwise'''  
    concaves = []
    for i in range(len(dist)):
        if curvature_maximos[i] == 0:
            concave = [0.0, 0.0]
        else:
            if i < (len(dist)-2):
                p_1 = curvature[i] - curvature[i+1]
                p_2 = curvature[i] - curvature[i+2]
                conva_convex = np.cross(p_1, p_2)
                if conva_convex > 0:
                    concave = curvature[i]
                else:
                    concave = [0.0, 0.0]
            elif i > (len(dist)-2):
                p_1 = curvature[i] - curvature[1]
                p_2 = curvature[i] - curvature[2]
                conva_convex = np.cross(p_1, p_2)
                if conva_convex > 0:
                    concave = curvature[i]
                else:
                    concave = [0.0, 0.0]
            elif i > (len(dist)-3):
                p_1 = curvature[i] - curvature[i+1]
                p_2 = curvature[i] - curvature[1]
                conva_convex = np.cross(p_1, p_2)
                if conva_convex > 0:
                    concave = curvature[i]
                else:
                    concave = [0.0, 0.0]            
        concaves.append(concave)  
    return concaves
    
    
def rotateTanget(curvature, tangents, angle=-np.pi/2):
    ''' Function that Rotate the tengent 90 degrees clockwise'''
    xx, yy = curvature[:,0], curvature[:,1]
    tx, ty = tangents[:,0], tangents[:,1]
    tx2 = tx - xx
    ty2 = ty - yy
    nx = tx2 * np.cos(angle) - ty2 * np.sin(angle)
    ny = tx2 * np.sin(angle) + ty2 * np.cos(angle)
    return np.array([nx + xx, ny + yy])
    

def getfricctionCone(curvature, nx, ny, alpha):
    '''Function that Rotate normal alpha degrees in two directions'''
    xx, yy = curvature[:,0], curvature[:,1]
    nx2 = nx - xx
    ny2 = ny - yy    
        
    nxa1 = nx2 * np.cos(alpha) - ny2 * np.sin(alpha)
    nya1 = nx2 * np.sin(alpha) + ny2 * np.cos(alpha)
    nxa1 = nxa1 + xx
    nya1 = nya1 + yy    

    nxa2 = nx2 * np.cos(-(alpha)) - ny2 * np.sin(-(alpha))
    nya2 = nx2 * np.sin(-(alpha)) + ny2 * np.cos(-(alpha))
    nxa2 = nxa2 + xx
    nya2 = nya2 + yy
    return np.array([nxa1, nya1]), np.array([nxa2, nya2])  
    
    
def changeLengthVector(tangents, curvature, length):
    tangents0 = tangents - curvature
    tangents1 = np.array([])
    for i in range(len(tangents0)):
        tangent = tangents0[i] / np.linalg.norm(tangents0[i])
        tangents1 = np.append(tangents1, tangent)
    tangents1 = tangents1.reshape(len(tangents0), 2)
    return (tangents1 * length) + curvature    
    
# Defining main functions-------------------------------------------
def plotfrictionCone(curvature, concaves, tangents, mu, plt1, length=30):
    '''Function that plots the friction cones'''
    alpha = m.atan(mu)
    #Getting the curvature
    xx, yy = curvature[:,0], curvature[:,1]
    #Getting the tangent
    tangents = changeLengthVector(tangents, curvature, length)
    nx_ny = rotateTanget(curvature, tangents)
    ##Rotate normal alpha degrees in two directions
    nxa1_nya1, nxa2_nya2 = getfricctionCone(curvature, nx_ny[0], nx_ny[1], alpha)
    for j in range(len(concaves)):
        if all(concaves[j]) != 0.0:
#            plt1.plot([xx[j], tangents[j][0]], [yy[j], tangents[j][1]], '--', linewidth = 1, color='red')
            plt1.plot([xx[j], nx_ny[0][j]], [yy[j], nx_ny[1][j]], '--', linewidth = 2, color='greenyellow')    
            plt1.plot([xx[j], nxa1_nya1[0][j]], [yy[j], nxa1_nya1[1][j]], '--', linewidth = 2, color='lightcoral')
            plt1.plot([xx[j], nxa2_nya2[0][j]], [yy[j], nxa2_nya2[1][j]], '--', linewidth = 2, color='lightcoral')
    return plt1    
        

def label2points(curvature, concaves, dist, curvature_maximos):
    '''Function that plotes the curvature and the points with maximun 
    curvaturte (convex and concave)'''
    fig,axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})
    # fig.suptitle("Curvature Maximization and Force Closure", fontsize = 'x-large')
# Plotting the maximos 
    axs[0].plot(dist)
    for i in range(len(curvature_maximos)):
        if curvature_maximos[i] != 0.0:
            axs[0].plot(i, curvature_maximos[i], "s", color='red')
        if all(concaves[i]) != 0.0:
            axs[0].plot(i, curvature_maximos[i], "s", color='orange', linewidth=3)
# Adding number to the maximons            
    xsd = np.linspace(0, len(curvature_maximos), len(curvature_maximos))
    ysd = curvature_maximos   
    i = 0
    for x,y in zip(xsd,ysd):
        if curvature_maximos[i] != 0.0:
            axs[0].annotate(i, (x,y), textcoords="offset points", 
                     xytext=(15,-5), ha='center', fontsize=14) 
        i += 1                
# Adding numbers to the curvature
    xs, ys = curvature[:,0], curvature[:,1]
    axs[1].scatter(xs, ys, s=0.5)
    axs[1].plot(xs, ys)    
    i = 0
    for x,y in zip(xs,ys):
        if curvature_maximos[i] != 0.0:
            axs[1].annotate(i, (x,y), textcoords="offset points",
                     xytext=(15,-5), ha='center', fontsize=14)  
        i += 1
# Adding points to the curvature        
    for j in range(len(concaves)):
        if curvature_maximos[j] != 0.0:
            axs[1].plot(curvature[j,0], curvature[j,1], "s", color='red')
        if all(concaves[j]) != 0.0:
            axs[1].plot(concaves[j][0], concaves[j][1], "s", color='orange', linewidth=3,) 
# Adding Legend             
    conca_patch = mpatches.Patch(color='orange', label='Concave')
    conve_patch = mpatches.Patch(color='red', label='Convex')
    axs[0].legend(handles=[conca_patch, conve_patch], prop={'size': 14}) 
#    axs[1].legend(handles=[conca_patch, conve_patch]) 
    axs[0].set_ylabel('Normal vector length')
    axs[0].set_xlabel('Point')
    axs[1].set_ylabel('y axis')
    axs[1].set_xlabel('x axis')
    return plt


def plotFC(fcPoints, curvature, concaves, tangents, mu, plt1, length=30):
    '''Function that plots the friction cones of the selected grasping points'''
    alpha = m.atan(mu)
    xx, yy = curvature[:,0], curvature[:,1]
    tangents = changeLengthVector(tangents, curvature, length)
    nx_ny = rotateTanget(curvature, tangents)
    nxa1_nya1, nxa2_nya2 = getfricctionCone(curvature, nx_ny[0], nx_ny[1], alpha)
    if fcPoints[0] is not None:
        for j in range(len(fcPoints)):     
            plt1.plot([xx[fcPoints[j]], nx_ny[0][fcPoints[j]]], [yy[fcPoints[j]], nx_ny[1][fcPoints[j]]], linewidth = 2, color='darkgreen', linestyle='--')    
            plt1.plot([xx[fcPoints[j]], nxa1_nya1[0][fcPoints[j]]], [yy[fcPoints[j]], nxa1_nya1[1][fcPoints[j]]], linewidth = 2, color='maroon', linestyle='--')
            plt1.plot([xx[fcPoints[j]], nxa2_nya2[0][fcPoints[j]]], [yy[fcPoints[j]], nxa2_nya2[1][fcPoints[j]]], linewidth = 2, color='maroon', linestyle='--')
    conca_patch = mpatches.Patch(color='orange', label='Concave')
    conve_patch = mpatches.Patch(color='red', label='Convex')
    orth = plt.Line2D((0,1),(0,0), color='greenyellow', linestyle='--', label='Orthogonal Line')
    fric = plt.Line2D((0,1),(0,0), color='lightcoral', linestyle='--', label='Friction Cone')
    orth_FC = plt.Line2D((0,1),(0,0), color='darkgreen', linestyle='--', label='Orthogonal Line in FC')
    fric_FC = plt.Line2D((0,1),(0,0), color='maroon', linestyle='--', label='Friction Cone in FC')
    plt1.legend(handles=[orth_FC, fric_FC, orth, fric, conca_patch, conve_patch], prop={'size': 14})
    return plt1
    

def getMaxConcaveCurvature(imbinary, order=4, num_points=300):
    '''Function that returns the maximun curvarure using EFD method, the
    maximun values and wether is Concave or not'''
    contours = getContours(imbinary)
    coeffs = getEFDCoeff(order, contours)
    curvature = reconstruct_contour(coeffs, num_points=num_points)
    tangents = getTangentCurvature(coeffs, num_points=num_points)      
    normals = getNormalsCurvature(coeffs, num_points=num_points)
    dist = getEuclideanNorm(normals, curvature)
    maximos, curvature_maximos = getMaxCurvatures(dist)
    concaves = getConcaves(curvature_maximos, dist, curvature)
    return contours, curvature, concaves, curvature_maximos, dist, tangents
    
    
#####################################################################
#####################################################################

###Getting the bigest sum curvature ---------------------------------
## Array of dist value where concave is true
def returnSumPoints(dist_where_concave):
    sums = []
    position = np.array([])
    for i in range(len(dist_where_concave)):
        for j in range(len(dist_where_concave)):
            if i != j:
                result = dist_where_concave[i] + dist_where_concave[j]
                if (result in sums) == False:
                    pos = [i,j]
                    position = np.append(position, pos, axis=0)
                    sums.append(result)  
    return sums, position.reshape(len(sums), 2)
    

def getSums(dist, concaves):
    dist_where_concave = []
    for i in range(len(dist)):
        if all(concaves[i]) != 0.0:
            dist_where_concave.append(dist[i])        
    sums, combinations = returnSumPoints(dist_where_concave)
    sumsMinMax = sorted(sums) 
    return dist_where_concave, combinations, sums


##Getting back the array index
def getCombinationBack(to_pick, combinations, dist_where_concave, dist):
    biggest_combination = combinations[to_pick[0]]
    dist1 = dist_where_concave[int(biggest_combination[0][0])]
    dist2 = dist_where_concave[int(biggest_combination[0][1])]
    distID1 = np.where(dist == dist1)
    distID2 = np.where(dist == dist2)
    return distID1, distID2


# Checking whether is force closure or not -------------------
def getforceClosure(distID1, distID2, curvature, tangents, concaves, mu):
    
    normals = rotateTanget(curvature, tangents)
    norm1 = [normals[0][int(distID1[0])], normals[1][int(distID1[0])]]
    norm2 = [normals[0][int(distID2[0])], normals[1][int(distID2[0])]]
    point1 = concaves[int(distID1[0])]
    point2 = concaves[int(distID2[0])]

    p1_n1 = norm1 - point1
    p1_p2 = point2 - point1
    unit_p1_n1 = p1_n1 / np.linalg.norm(p1_n1)
    unit_p1_p2 = p1_p2 / np.linalg.norm(p1_p2)
    dot_product1 = np.dot(unit_p1_n1, unit_p1_p2)
    angle1 = np.arccos(dot_product1)

    p2_n2 = norm2 - point2
    p2_p1 = point1 - point2
    unit_p2_n2 = p2_n2 / np.linalg.norm(p2_n2)
    unit_p2_p1= p2_p1 / np.linalg.norm(p2_p1)
    dot_product2 = np.dot(unit_p2_n2, unit_p2_p1)
    angle2 = np.arccos(dot_product2)     
    
    alpha = m.atan(mu)
    if angle1 <= alpha and angle2 <= alpha:
        return True
    else:
        return False


def getFCpoints(dist, curvature, tangents, concaves, mu):
    dist_where_concave, combinations, sums = getSums(dist, concaves)
    sumsMinMax = sorted(sums)
    k = 1
    for k in range(len(sums)):
        to_pick = np.where(sums == sumsMinMax[-k])
        distID1, distID2 = getCombinationBack(to_pick, combinations, dist_where_concave, dist) 
        force_closure = getforceClosure(distID1, distID2, curvature, tangents, concaves, mu)
        if force_closure == True:
#            print('----FORCE CLOSURE FOUND----')
            return np.array([distID1[0], distID2[0]])
            break
        else:
            continue
#    print('----NO FORCE CLOSURE FOUND----')
    return np.array([None, None])   
    
    
def getGraspingPoints(concaves, fcPoints, contours):
    contours_mean = getContoursMean(contours)
    contours = contours[0].reshape((contours[0].shape[0],2))
    p1 = concaves[int(fcPoints[0])]
    p2 = concaves[int(fcPoints[1])]
    p1[0] = p1[0] + contours_mean[0]
    p1[1] = p1[1] + contours_mean[1]
    p2[0] = p2[0] + contours_mean[0]
    p2[1] = p2[1] + contours_mean[1]
    p1, p2 = np.array([p1]), np.array([p2])
    index1 = directed_hausdorff(p1, contours)[1:] # get the closest one
    index2 = directed_hausdorff(p2, contours)[1:]
    closestP1 = np.array([contours[:,0][index1[1]], contours[:,1][index1[1]]])
    closestP2 = np.array([contours[:,0][index2[1]], contours[:,1][index2[1]]])
    middle = (closestP1 + closestP2) / 2
    return np.array([closestP1, closestP2, middle])
  

def getWidth(grasping_points_3d):
    p1 = grasping_points_3d[0]
    p2 = grasping_points_3d[1]
    width = m.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
    return width
#####################################################################
#####################################################################














