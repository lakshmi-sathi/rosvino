# rosvino
This repository contains the ROS package 'rosvino'. It uses the Intel Inference Engine API to run inference on the Intel Neural Compute Stick (Movidius VPU). This package can be used on linux-capable embedded systems and robots that utilize ROS, to offload and run inference on the Intel Neural Compute Stick.

Kindly cite this Repository if you use this package

# Intel Inference Engine API
![image](https://user-images.githubusercontent.com/58559090/137576074-15a97e86-e771-4be4-acbd-2e40e91ee771.png)

# Evaluation-of-the-Intel-Movidius-VPU-Neural-Compute-Stick

The basic flow for using InferenceEngine API for inference is:

![image](https://user-images.githubusercontent.com/58559090/137576183-fc728405-ee3b-4cfb-8bc3-17432beba274.png)

The ROS nodes are based on the latest OpenVINO, they are - detect.cpp and segment.cpp. Using these two nodes, it is attempted to run and analyse 7 deep learning models each for detection task and segmentation task. In both nodes the inference is done in a blocking manner (not pipelined) so only after completion of an inference, the control flow can continue.

In the ‘detect’ node the resulting detections are published in the form of a message object, one by one, and no image is published at the end. While in the ‘segment’ node, the output is collected in the form of a class matrix (class of each pixel) and a probability matrix (probability of each pixel belonging to that class) which is then processed for visualisation, and published.

The launch files are placed in the ‘launch’ folder (all media files are placed in the ‘input’ folder) and there is a launch file for each model. Each launch file name starts with the task (detection/segmentation) followed by a short name for the model used. For example, ‘detection_pedestrian.launch’, ‘segmentation_road.launch’.
Each launch file has following arguments:
‘mode’ - input mode, whether to input image or video
‘input_image’ - name of input image to provide repeatedly (from ‘input’ folder)
‘input_video’ - name of input video (from ‘input’ folder)
‘device’ - name of device to run the inference on.
The default mode is ‘image’, input_image is ‘road7.jpg’, input_video is ‘vid2.mp4’ and device is ‘MYRIAD’. Here are a few examples of usage:

> roslaunch  rosvino  detection_person_retail.launch  device:=CPU  | grep “Latency”

This uses the model ‘person-detection-retail-0002’ to run inference on ‘CPU’ where the mode is by default ‘image’ and the default input_image ‘road7.jpg’ is taken (from ‘input’ folder). The grep on “Latency” prints out just the average latency alone without printing out the detections.

> roslaunch rosvino segmentation_hrnet.launch mode:=video  input_video:=video.mp4

This runs the hrnet segmentation model by default on ‘MYRIAD’ in ‘video’ mode with the given input_video of the name ‘video.mp4’ (taken from ‘input’ folder)

> roslaunch rosvino segmentation_road.launch device:=CPU mode:=video

This runs the ‘road-segmentation-adas-0001’ model on ‘CPU’ in ‘video’ mode with the default input_video ‘vid2.mp4’ (taken from ‘input’ folder)


The detailed raw observations made are collected in this spreadsheet. A variety of models are chosen, some are converted into IR after download, others are downloaded as IR itself. The GFLOPs (Giga FLoating point OPerations) or the complexity, amount of computation required by the models vary from as low as 1.5 x 109 to as high as 364 x 109 floating point operations. The MParams or the parameter count varies from 0.184 x 106 to 66 x 106.  

We can quickly observe some valuable patterns that we were looking forward to but hadn’t observed in earlier attempts:
* Across all the models, the turtlebot CPU rarely gives above 1 frame per second and in comparison we can clearly see the larger numbers on the VPU side. If we take the case of detection models, we can see that the VPU is able to give about 2 - 3 times the throughput obtained on running the same model on CPU. 
* It is observed that the average latencies obtained when running a model on VPU is more consistent compared to running on CPU. This is understood as being due to the fact that the CPU is handling many more processes and not just the inference alone. So the latency measurements while running on CPU are having a higher variance.
* If we compare the CPU Loads, we can see a minimum 2 times reduction in CPU Load if inference is offloaded to the VPU and upto 4 times.
* Consider the latency while running inference on the VPU, in case of detection models, we can see a 2.3 to 3 times improvement in latency. However, there is more to the story, if we observe the unet segmentation model which is much more complex (260GFLOPs compared to ~5GFLOPs) we can see a remarkable difference in performance between CPU and VPU runs. The VPU latency was just 2.65 secs when the CPU took 20 secs. That’s an 8 to 9 times improvement! This indicates that the improvement in performance becomes very significant when we deal with larger models.
* Despite the greater advantage when working with larger models, there is the limitation that only 500MB of RAM is available onboard the compute stick and that prevents loading of models with more than about 60 MParams.
* By looking through both the CPU section and VPU section, we can notice yet another thing, there are more cases of compatibility issues like some layer not supported, issue while creating graph on device or some such issue when it comes to the VPU. While for the CPU, most of the models are supported without issues. CPU also supports ‘quantised models’ that use 8bit integer precision which could significantly improve performance and energy cost. These models are not supported by the VPU (only works with 16bit floating point precision)
* Looking at things from the real-time inference perspective. We can see the best throughput obtained is 13 frames per second by the VPU which is quite far from the common 24 fps used for smooth viewing experience in videos. 
* However, it should be noted that the VPU would use about 1-2 Watts when the turtlebot CPU would use 10 Watts or above for running the inferences.

Putting together these observations, we can observe that the Intel Movidius Neural Compute Stick is doing much more than just offloading inference from the CPU, however, though it offloads the inference as well as significantly boosts the inference performance, it is still far from being real-time. For achieving that, pipelining of inferences and use of a combination of two or more Neural Compute Sticks would be some things to consider.

