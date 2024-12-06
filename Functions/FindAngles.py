"""
Author: Luis Angel Ponce Pacheco
====================

Function that returns the Rotation Matrix of a bunch of 3D points

"""
# Importing some libraries------------------------------------- 
from skspatial.objects import Plane
import numpy as np
import math as m
from scipy.spatial.transform import Rotation as R

# Defining some functions-------------------------------------------
def distance(point1, point2):
    '''Function that returns absolute distance between two points'''
    d = m.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    return d
   
    
def atan2PositiveOnly(point):
    '''Function that returns the angle beween positive X axes and a point'''
    if point[1] >= 0:
        angle = m.atan2(point[1], point[0])
    else: 
        angle = 2*m.pi + m.atan2(point[1], point[0])
    return angle
 
        
def getAngleSign(keypoints):
    '''Function that returns the angle between line 3-2 and 3-4. 
    Positive values -> Conter-clockwise.
    Negative values -> Clockwise'''
    keypoints = keypoints - keypoints[2]
    angleP2 = atan2PositiveOnly(keypoints[1])
    angleP4 = atan2PositiveOnly(keypoints[3])
    angleP4hat = angleP4 - angleP2
    d4 = distance(keypoints[2], keypoints[3])
    x4hat = d4 * m.cos(angleP4hat)
    y4hat = d4 * m.sin(angleP4hat)
    point4hat = np.array([x4hat, y4hat])
    return m.atan2(point4hat[1], point4hat[0])

    
def angleVectors(v1, v2):
    '''Function that return the minimun angle between two vectors'''
    dot_products = np.einsum("ij,ij->i", v1.reshape(-1, 3), v2.reshape(-1, 3))
    cosines = dot_products / np.linalg.norm(v1) / np.linalg.norm(v2)
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    return angles[0] if v1.ndim == 1 and v2.ndim == 1 else angles


def getPlane(raw_points):
    '''Function that return a plane given a bunch of points'''    
    centroid_ = np.mean(raw_points, axis=0)
    points_centered = raw_points - centroid_
    try:
        u, _, _ = np.linalg.svd(points_centered.T)
    except:
        print('----Using Defaul Plane---')
        u = np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,1.0]])
    normal = u[:, 2]
    plane = Plane(centroid_, normal)
    return plane

    
def getPointsInPlane(raw_points, plane):
    '''Function that return points projected in a given plane'''        
    points_in_plane = []
    for i in range(len(raw_points)):
        point_in_plane = plane.project_point(raw_points[i])
        points_in_plane.append(point_in_plane) 
    return points_in_plane


def selectVectors(points, center):
    '''Function that return the vector to calculate Z direction given the center point'''
    if center == 0:
        v1, v2 = points[3], points[2]
    elif center == 1:
        v1, v2 = points[3], points[2]
    elif center == 2:
        v1, v2 = points[1], points[3]
    elif center == 3:
        v1, v2 = points[2], points[0]
    elif center == 4:
        v1, v2 = points[2], points[3]
    return v1, v2


def selectZ (clase, angleSing, v1, v2):
    '''Function that gives the direction of Z'''
    # Top Right 			#for two classes detection only	#for five classes detection 
    if clase == 1 and angleSing >= 0: #clase == 0				#clase == 1
        z_object = np.cross(v2, v1)
    # Top Left
    elif clase == 1 and angleSing < 0: #clase == 0				#clase == 1
        z_object = np.cross(v1, v2)   
    # Bottom Left
    elif clase == 3 and angleSing >= 0: #clase == 1				#clase == 3
        z_object = np.cross(v2, v1) #np.cross(v1, v2) -> Point down
    # Bottom Right
    elif clase == 3 and angleSing < 0: #clase == 1				#clase == 3
        z_object = np.cross(v1, v2) #np.cross(v2, v1) -> Point down
    else:
        print('----NO Z AXIS FOUND----')
        print('----pointing Z up----')
        z_object = np.array([0.0, 0.0, -100.0])
    return z_object


def selectX(points, center):
    '''Function that return the direction of X given the center'''
    if center == 0:
        x_object = points[1]
    elif center == 1:
        x_object = points[0]
    elif center == 2:
        x_object = points[1]
    elif center == 3:
        x_object = points[2]
    elif center == 4:
        x_object = points[2]
    return x_object  


# Defining main functions-------------------------------------------
def getRotationMatrix(raw_points, center, clase):
    '''Functiuon that returns the rotation matrix of a bunch of points.
    Z pointing perpendicular to the average poits plane, X pointing to 
    one of the points, and Y is given the right hand rule'''
    reference_frame = np.array([[1,0,0], [0,1,0], [0,0,1]])
    raw_points = raw_points - raw_points[center]
    
    plane = getPlane(raw_points)    
    points_in_plane = getPointsInPlane(raw_points, plane)    
    points_in_plane = points_in_plane - points_in_plane[center]    
    angleSing = getAngleSign(points_in_plane)
    
    v1, v2 = selectVectors(points_in_plane, center)
    z_object = selectZ(clase, angleSing, v1, v2)
    x_object = selectX(points_in_plane, center)
    y_object = np.cross(z_object, x_object)
    
    Q11 = m.cos(angleVectors(reference_frame[0], x_object))
    Q12 = m.cos(angleVectors(reference_frame[0], y_object))
    Q13 = m.cos(angleVectors(reference_frame[0], z_object))

    Q21 = m.cos(angleVectors(reference_frame[1], x_object))
    Q22 = m.cos(angleVectors(reference_frame[1], y_object))
    Q23 = m.cos(angleVectors(reference_frame[1], z_object))

    Q31 = m.cos(angleVectors(reference_frame[2], x_object))
    Q32 = m.cos(angleVectors(reference_frame[2], y_object))
    Q33 = m.cos(angleVectors(reference_frame[2], z_object))

    rotation_matrix = np.array([[Q11, Q12, Q13], [Q21, Q22, Q23], [Q31, Q32, Q33]])
    return rotation_matrix


def getAllAngles(kp3d, clase):
    '''Function that returns the rotation angles of all the keyPoints'''
    anglesAllPoints =  np.zeros([1, 3])
    for i in range(len(kp3d)):
        rotation_matrix = getRotationMatrix(kp3d, center=i, clase=clase)
        r = R.from_matrix(rotation_matrix)
        angles = np.array([r.as_euler('xyz', degrees = True)])
        anglesAllPoints = np.append(anglesAllPoints, angles, axis=0)
    anglesAllPoints = np.delete(anglesAllPoints, 0, 0)
    return anglesAllPoints
    

#########################################################################
#########################################################################

#########################################################################
#########################################################################
# Pose from two points in the contour
 
    
    
def getRotatMatrixFrom2Points(raw_points, p1_p2, clase):
    '''Functiuon that returns the rotation matrix given two points in the 
    contour and the keypoints.
    Z pointing perpendicular to the average poits plane, Y pointing to 
    the second point , and X is given the right hand rule'''
    center = 1
    reference_frame = np.array([[1,0,0], [0,1,0], [0,0,1]])
    p1_p2 = p1_p2 - raw_points[center]
    raw_points = raw_points - raw_points[center]
        
    plane = getPlane(raw_points)   
    points_in_plane = getPointsInPlane(raw_points, plane)    
    points_in_plane = points_in_plane - points_in_plane[center]    
    angleSing = getAngleSign(points_in_plane)
    
    p1_p2_in_plan = getPointsInPlane(p1_p2, plane)
    p1_p2_in_plan = p1_p2_in_plan - points_in_plane[center]
    
    v1, v2 = selectVectors(points_in_plane, center)
    z_object = selectZ(clase, angleSing, v1, v2)
    y_object = p1_p2_in_plan[1] - p1_p2_in_plan[2]     
    x_object = np.cross(y_object, z_object)
        
    Q11 = m.cos(angleVectors(reference_frame[0], x_object))
    Q12 = m.cos(angleVectors(reference_frame[0], y_object))
    Q13 = m.cos(angleVectors(reference_frame[0], z_object))

    Q21 = m.cos(angleVectors(reference_frame[1], x_object))
    Q22 = m.cos(angleVectors(reference_frame[1], y_object))
    Q23 = m.cos(angleVectors(reference_frame[1], z_object))

    Q31 = m.cos(angleVectors(reference_frame[2], x_object))
    Q32 = m.cos(angleVectors(reference_frame[2], y_object))
    Q33 = m.cos(angleVectors(reference_frame[2], z_object))

    rotation_matrix = np.array([[Q11, Q12, Q13], [Q21, Q22, Q23], [Q31, Q32, Q33]])
    return rotation_matrix


def getSide(raw_points, clase):
    '''Function that returns returns 'rigth' or 'left' depending on the chicken leg side'''
    center = 1
    raw_points = raw_points - raw_points[center]
    plane = getPlane(raw_points)  
    points_in_plane = getPointsInPlane(raw_points, plane)    
    points_in_plane = points_in_plane - points_in_plane[center]    
    angleSing = getAngleSign(points_in_plane)
    # Top Right
    if clase == 1 and angleSing >= 0: #clase == 0				#clase == 1
        side = 'Top Right'
    # Top Left
    elif clase == 1 and angleSing < 0: #clase == 0				#clase == 1
        side = 'Top Left'  
    # Bottom Left
    elif clase == 3 and angleSing >= 0: #clase == 1				#clase == 3
        side = 'Bottom Left'
    # Bottom Right
    elif clase == 3 and angleSing < 0: #clase == 1				#clase == 3
        side = 'Bottom Right'
    
    elif clase == 0:
        side = 'Part Top'
    elif clase == 2:
        side = 'Part Bottom'
    elif clase == 4:
        side = 'Tilted'
    else:
        side = 'No Whole Piece'
    return side    
    
    
