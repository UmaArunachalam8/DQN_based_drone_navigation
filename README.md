# DQN_based_drone_navigation
A technique combining behavioral cloning and DQN for trail following.
## Prerequisites and Setup
COMPATIBLITY 

All the codes run on Ubuntu 16.04 with python 3.6.8 and Unreal 4.18. Works on lower python3 versions as well, BUT NOT HIGHER.

LANDSCAPE MOUNTAIN ENVIRONMENT

Our entire project is tested in the simulated environment called landscape mountains, which is one of the very few open source - free environments compatible with AirSim. We have simplified the process of cross compiling with linux and created a readily available easy to use plug-in for this environment

1. Copy the LandscapeMountains 4.18 - 2 folder into your required destination.
2. To use this plugin, open the LandscapeMountains.uproject in Unreal editor

DEPENDENCIES

Running as administrator ( if using windows ) or from terminal, pip install the following packages
1. Jupyter
2. Matplotlib
3. Image
4. Keras, Keras_tqdm
5. OpenCv
6. Msgpack-rpc-python
7. Pandas
8. Numpy
9. Scipy
10. Cntk

To execute any code, make sure the mode of execution is Multirotor in Settings.json ( Will be available in Documets/AirSim after build ) and start the unreal engine and open the required environment, as in our case - Landscape Mountains

## Behavioral Cloning

To implement this technique follow the steps mentioned below :
1. Data Exploration and preparation

Play with the simulator to collect training data, the corresponding pitch, yaw, and roll values, along with other state parameters such as velocity in the x, y, z directions should be saved along with the input from forward facing cam of the drone.

Create two empty folders called models and cooked_data in the same working directory

Run the cook_data.py code 

2. Training
Run the train_model.py code

3. Testing
Run the fly_model.py code

## Reinforcement Learning
1. A one step, data processing and training code is written for ease of use. The agent can be trained by running the dqn_fyp_f3.py. This step might take more than two days to achieve best performance 

2. View the results by logging into tensor board using the following command

  `tensorboard --logdir path_to_your_metric_folder` 



