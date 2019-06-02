#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 10:20:05 2019

@author: uma
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.models import load_model
import sys
import time
import numpy as np
import setup_path
import airsim
import glob
import math
import keras.backend as K
from keras.preprocessing import image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

if ('../../PythonClient/' not in sys.path):
    sys.path.insert(0, '../../PythonClient/')

MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('models/models/*.h5') 
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model
    
print('Using model {0} for testing.'.format(MODEL_PATH))

model = load_model(MODEL_PATH)
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
#client.moveToPositionAsync(-10, 10, -10, 5).join()

#client.moveToPositionAsync(0, 0, 0, 5).join()
#client.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
#time.sleep(0.5)

image_buf = np.zeros((1, 66, 200, 3))

def get_image():
    
    image_response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    return image_rgba[78:144,27:227,0:3].astype(float)

zc = client.getMultirotorState().kinematics_estimated.position.z_val - 6
xc = client.getMultirotorState().kinematics_estimated.position.x_val 
yc = client.getMultirotorState().kinematics_estimated.position.y_val
client.moveToPositionAsync(xc, yc, zc, 2).join()
duration = 0.2
alpha = 0.5
beta = 0.5
r = 0.3
print(xc,yc)
while (True):
    client.enableApiControl(True)
    client.armDisarm(True)
    image_buf[0] = get_image()
    image_buf[0] /= 255
    p = client.getMultirotorState().kinematics_estimated.position
    q = client.getMultirotorState().kinematics_estimated.orientation
    x = p.x_val
    y = p.y_val
    #state_buf[0] = np.array([0,0,0,0])
    #state_buf[0] = np.array([q.w_val, p.x_val, p.y_val, p.z_val])
    #model_output = model.predict([image_buf, state_buf])
    model_output = model.predict([image_buf])
    steering = round((0.82*(float((model_output[0][0]*2.0)-1))), 2)
    client.rotateToYawAsync(steering).join()
    #client.moveToPositionAsync(xn, yn, zc, 1).join()
    #steering = 
    yaw = beta * 3.14 * 0.5 * steering
    #xn = xc + r * math.cos(steering)
    #yn = yc + r * math.sin(steering)
    #model_output = [[0.523079, -0.018476, -0.01134767, -0.85174067],[0, 0, 0, 0]]
    #controls = toEulerianAngle(model_output[0])
    client.moveToPositionAsync(x - 0.3, y - 0.3, zc, 1).join()
    #print(xn,yn)
    #xc = xn
    #yc = yn
    #client.rotateToYawAsync(yaw).join()
    #client.moveByVelocityAsync(1, 1, 0, 2).join()
    #client.moveByAngleZAsync(controls[0], controls[1], z, 0, duration).join()
    #client.moveByAngleThrottleAsync(controls[0], 0, 1, controls[2], duration).join()
    #client.moveByRC(rcdata = airsim.RCData(pitch = controls[0], throttle = 1.0, is_initialized = True, is_valid = True))
