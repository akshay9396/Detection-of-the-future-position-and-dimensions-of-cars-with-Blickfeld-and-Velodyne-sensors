#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np # load point clouds and more array/vector operations\n",
import cv2 # load images\n",
import plotly.graph_objects as go # visualize point clouds
from math import *
import matplotlib.pyplot as plot
from plotly.subplots import make_subplots
import math


# In[35]:


root = "/Users/shrinidhimeti/Desktop/Lidar and Radar/dataset/Record1/"


# In[36]:


def inv_rot(rot):
    """
        Calculates for a given 4x4 transformation matrix (R|t) the inverse.
    """
    inv_rot_mat = np.zeros((4,4))
    inv_rot_mat[0:3,0:3] = rot[0:3,0:3]
    inv_rot_mat[0:3,3] = -np.dot(rot[0:3,0:3].T, rot[0:3,3])
    inv_rot_mat[3,3] = 1
    return inv_rot_mat


# In[37]:


def transfer_points(points, rot_t):
    """
        Calculates the transformation of a point cloud for a given transformation matrix.
    """
    points = np.concatenate([points, np.ones([1,points.shape[1]])])
    points[0:3,:] = np.dot(rot_t, points[0:4,:])[0:3, :]
    return points[0:3]


# In[38]:


def make_boundingbox(label):
    """
        Calculates the Corners of a bounding box from the parameters.
    """
    corner = np.array([
        [- label[3]/2, - label[4]/2, - label[5]/2],
        [- label[3]/2, - label[4]/2, + label[5]/2],
        [- label[3]/2, + label[4]/2, + label[5]/2],
        [+ label[3]/2, + label[4]/2, + label[5]/2],
        [+ label[3]/2, + label[4]/2, - label[5]/2],
        [- label[3]/2, + label[4]/2, - label[5]/2],
        [- label[3]/2, - label[4]/2, - label[5]/2],
        [+ label[3]/2, - label[4]/2, - label[5]/2],
        [+ label[3]/2, - label[4]/2, + label[5]/2],
        [- label[3]/2, - label[4]/2, + label[5]/2],
        [- label[3]/2, + label[4]/2, + label[5]/2],
        [- label[3]/2, + label[4]/2, - label[5]/2],
        [+ label[3]/2, + label[4]/2, - label[5]/2],
        [+ label[3]/2, - label[4]/2, - label[5]/2],
        [+ label[3]/2, - label[4]/2, + label[5]/2],
        [+ label[3]/2, + label[4]/2, + label[5]/2],
    ])
    corner = transfer_points(corner.T, rt_matrix(yaw = label[6])).T
    corner = corner + label[0:3]
    #print(corner[:,0:1]) [:,1:2]
    #print(transfer_points)
    return corner


# In[39]:


def rt_matrix(x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
    """
        Calculates a 4x4 Transformation Matrix. Angels in radian!
    """
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    
    rot = np.dot(np.dot(np.array([[c_y,-s_y,0],
                                     [s_y,c_y,0],
                                     [0,0,1]]),
                           np.array([[c_p,0,s_p],
                                     [0,1,0],
                                     [-s_p,0,c_p]])),
                            np.array([[1,0,0],
                                     [0,c_r,-s_r],
                                     [0,s_r,c_r]]))
    matrix = np.array([[0,0,0,x],
                     [0,0,0,y],
                     [0,0,0,z],
                     [0,0,0,1.]])
    matrix[0:3,0:3] += rot
    #print(matrix)
    return matrix


# In[40]:


rotationmat = np.load(root + "calibrationMat.npy")
invrotationmat = inv_rot(rotationmat)


# In[41]:


ind = 29
blick = np.loadtxt(root + "Blickfeld/point_cloud/%06d.csv" % ind)
velo = np.loadtxt(root + "Velodyne/point_cloud/%06d.csv" % ind)
velo = transfer_points(velo.T[0:3], rotationmat).T

bb = np.loadtxt(root + "/Blickfeld/bounding_box/%06d.csv" % ind)
bb = make_boundingbox(bb)
#print(bb[:,:1])
#print(bb)


# In[42]:


ind = 28
blick = np.loadtxt(root + "Blickfeld/point_cloud/%06d.csv" % ind)
velo = np.loadtxt(root + "Velodyne/point_cloud/%06d.csv" % ind)
velo = transfer_points(velo.T[0:3], rotationmat).T

bb1 = np.loadtxt(root + "/Blickfeld/bounding_box/%06d.csv" % ind)
bb1 = make_boundingbox(bb1)
#print(bb[:,:1])


# In[45]:


ind = 30
blick = np.loadtxt(root + "Blickfeld/point_cloud/%06d.csv" % ind)
velo = np.loadtxt(root + "Velodyne/point_cloud/%06d.csv" % ind)
velo = transfer_points(velo.T[0:3], rotationmat).T

bb2 = np.loadtxt(root + "/Blickfeld/bounding_box/%06d.csv" % ind)
bb2 = make_boundingbox(bb2)
#print(bb[:,:1])
print(bb2)


# In[44]:


C = []
for i in range(len(bb1)):
    difference = bb[i] - bb1[i]
    C.append(difference)
print("Difference")
for i in range(len(C)):
    print(C[i])
C1=[]
print("Position Predection for the next image")
for i in range(len(bb)):
    addition = bb[i] + C[i]
    C1.append(addition)
for i in range(len(C1)):
    print(C1[i])
C2=[]
print("error")
for i in range(len(bb2)):
    error = C1[i] - bb2[i]
    C2.append(error)
for i in range(len(C2)):
    print(C2[i]) 
print("Prediction")
my_array1 = np.array(C1)
print(my_array1)


# In[47]:


data = [go.Scatter3d(x = blick[:,0],
                     y = blick[:,1],
                     z = blick[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(blick.shape[0]),
                    marker={
                        'size': 2,
                        'color': "blue",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = velo[:,0],
                     y = velo[:,1],
                     z = velo[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(velo.shape[0]),
                    marker={
                        'size': 2,
                        'color': "green",
                        'colorscale':'rainbow',
}),
    go.Scatter3d(x = bb2[:,0],
                     y = bb2[:,1],
                     z = bb2[:,2],                     
                     text=np.arange(1),
                    mode='lines+markers',
                    marker={
                        'size': 3,
                        'color': "red",
                        'colorscale':'rainbow',       
}),   
    
    go.Scatter3d(x = my_array1[:,0],
                     y = my_array1[:,1],
                     z = my_array1[:,2],                     
                     text=np.arange(1),
                    mode='lines+markers',
                    marker={
                        'size': 3,
                        'color': "green",
                        'colorscale':'rainbow',       
})    
]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-25, 25], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [-7, 28], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-12.5, 12.5], 'rangemode': 'tozero'}
    }
)
go.Figure(data=data, layout = layout)


# In[14]:


ind = 2
blick = np.loadtxt(root + "Blickfeld/point_cloud/%06d.csv" % ind)
velo = np.loadtxt(root + "Velodyne/point_cloud/%06d.csv" % ind)
blick = transfer_points(blick.T[0:3], invrotationmat).T

bb = np.loadtxt(root + "/Velodyne/bounding_box/%06d.csv" % ind)
bb = make_boundingbox(bb)


# In[15]:


data = [go.Scatter3d(x = blick[:,0],
                     y = blick[:,1],
                     z = blick[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(blick.shape[0]),
                    marker={
                        'size': 2,
                        'color': "blue",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = velo[:,0],
                     y = velo[:,1],
                     z = velo[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(velo.shape[0]),
                    marker={
                        'size': 2,
                        'color': "green",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = bb[:,0],
                     y = bb[:,1],
                     z = bb[:,2],
                     text=np.arange(8),
                    mode='lines+ markers', type='scatter3d',
                    marker={
                        'size': 2,
                        'color': "red",
                        'colorscale':'rainbow',
})
]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-25, 25], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [-25, 28], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-12.5, 12.5], 'rangemode': 'tozero'}
    }
)
go.Figure(data=data, layout = layout)


# In[25]:


root = "/Users/shrinidhimeti/Desktop/Lidar and Radar/dataset/Record2/"


# In[26]:


rotationmat = np.load(root + "calibrationMat.npy")
invrotationmat = inv_rot(rotationmat)


# In[27]:


ind = 7
blick = np.loadtxt(root + "Blickfeld/point_cloud/%06d.csv" % ind,)
velo = np.loadtxt(root + "Velodyne/point_cloud/%06d.csv" % ind)
velo = transfer_points(velo.T[0:3], rotationmat).T

bb = np.loadtxt(root + "/Blickfeld/bounding_box/%06d.csv" % ind)
if len(bb):
    bb = make_boundingbox(bb)
else:
    bb = np.array([[None,None,None,None,None,None,None]])


# In[31]:


data = [go.Scatter3d(x = blick[:,0],
                     y = blick[:,1],
                     z = blick[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(blick.shape[0]),
                    marker={
                        'size': 2,
                        'color': "blue",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = velo[:,0],
                     y = velo[:,1],
                     z = velo[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(velo.shape[0]),
                    marker={
                        'size': 2,
                        'color': "green",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = bb2[:,0],
                     y = bb2[:,1],
                     z = bb2[:,2],
                     text=np.arange(8),
                    mode='lines+ markers', type='scatter3d',
                    marker={
                        'size': 2,
                        'color': "red",
                        'colorscale':'rainbow',
})
     
]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-25, 25], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [-25, 25], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-12.5, 12.5], 'rangemode': 'tozero'}
    }
)
go.Figure(data=data, layout = layout)


# In[23]:


ind = 8
blick = np.loadtxt(root + "Blickfeld/point_cloud/%06d.csv" % ind,)
velo = np.loadtxt(root + "Velodyne/point_cloud/%06d.csv" % ind)
blick = transfer_points(blick.T[0:3], invrotationmat).T

bb = np.loadtxt(root + "/Velodyne/bounding_box/%06d.csv" % ind)
if len(bb):
    bb = make_boundingbox(bb)
else:
    bb = np.array([[None,None,None,None,None,None,None]])


# In[24]:


data = [go.Scatter3d(x = blick[:,0],
                     y = blick[:,1],
                     z = blick[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(blick.shape[0]),
                    marker={
                        'size': 2,
                        'color': "blue",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = velo[:,0],
                     y = velo[:,1],
                     z = velo[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(velo.shape[0]),
                    marker={
                        'size': 2,
                        'color': "green",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = bb[:,0],
                     y = bb[:,1],
                     z = bb[:,2],
                     text=np.arange(8),
                    mode='markers', type='scatter3d',
                    marker={
                        'size': 2,
                        'color': "red",
                        'colorscale':'rainbow',
})
]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-25, 25], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [-25, 25], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-25, 25], 'rangemode': 'tozero'}
    }
)
go.Figure(data=data, layout = layout)


# In[11]:


C = []
for i in range(len(bb1)):
    difference = bb[i] - bb1[i]
    C.append(difference)
    addition = bb[i] + difference
    C.append(addition)
    error = addition - bb2[i]
    C.append(error)
my_array = np.array(C)
print("Difference")
i=0
while i < len(my_array):
    print(my_array.item(i,0),my_array.item(i,1),my_array.item(i,2))
    i=i+3
print("Predected Values")
i=1
while i < len(my_array):
    print(my_array.item(i,0),my_array.item(i,1),my_array.item(i,2))
    i=i+3
print("Error")
i=2
while i < len(my_array):
    print(my_array.item(i,0),my_array.item(i,1),my_array.item(i,2))
    i=i+3

