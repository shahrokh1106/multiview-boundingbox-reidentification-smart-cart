# Multi-view multi-object bounding-box re-identification in a three-camera system installed in a shopping cart

* Dataset: A dataset of images captured by a three-camera system installed inside a smart shopping
* Camera Calibration: Implementing the Tsai method for camera calibration, ensuring accurate geometric calibration of the camera system to correct lens distortion and intrinsic parameters
* Epipolar Geometry Pipeline: Developing an epipolar geometry pipeline based on fundamental and essential matrices
* 3D Localization Estimation: Designing and implementing an approach for 3D localization estimation without computing a depth map, leveraging geometric constraints and feature triangulation techniques
* Siamese Network Training: Modeling and training a Siamese neural network architecture based on triplet loss for bounding box re-identification purposes, enabling robust and discriminative feature embeddings
* End-to-end Multi-View Bounding Box Re-Identification algorithm: Developing a multi-object multi-camera bounding box re-identification algorithm, integrating the trained Siamese network with epipolar constraints to enable accurate and efficient object re-identification across different camera views
* Deployment Automation: Implementing deployment automation solutions using C++ for Edge-TPU Coral and Raspberry Pi platforms, optimizing inference performance and resource utilization for real-time application deployment at the edge.

The main algorithm iterates all possible triple combinations of bounding boxes to find a set of bounding boxes with minimum cost. Such a cost is computed by epipolar geometry constraint and cosine similarity between feature embedding belonging to the bounding boxes. <br/>
* Epipolar constraint: Given the center of a bounding box in one view, the center of the matched bounding box in the other way is expected to be on the corresponding epipolar line. The cost is computed based on the distance from the lines.
* Embedding constraint: Given a bounding box in from a camera view, its feature embedding could be captured in two ways: 1) getting the corresponding features from the backbone of MobileNet, 2) getting the corresponding features from a trained model. The current setup is based on the second approach.
  * h

A sample result from the main  multi-view multi-object bounding-box re-identification algorithm in the presence of noise (inaccurate bounding boxes detected by an object detection model):

![results](https://github.com/shahrokh1106/multiview-boundingbox-reidentification-smart-cart/assets/44213732/bbae13c3-cb97-45a1-afea-bcfb479d0ecd)

An example of using epipolar constraint when the bounding boxes are matched vs when we have unmatched bounding boxes the main  multi-view multi-object bounding-box re-identification algorithm:
![epipolar](https://github.com/shahrokh1106/multiview-boundingbox-reidentification-smart-cart/assets/44213732/e8805a16-3578-4bf1-ba91-25990e6bba99)

To run the main algorithm on a sample from the dataset, run "BBoxMatchingMobNet.py", where

* *--height* is height of the input images which must match the calibration setup (320)
* *--width* is width of the input images which must match the calibration setup (320)
* *--eppipolar* determines if the algorithm should use eppipolar constraints or not 
* *--embedding* determines if the algorithm should use the pretrained embedding model or not 
* *--embedpath* determines the path of the embedding model trained based on an triplet-loss function on a dataset of triple images (anchor, positive, and negative)
* *--calibpath* is a list of three string paths that indicates the location of json calib files for cam0, cam1, and cam2 respectively
* *--boxscaling* is an integer that indicates the required scaling when getting a bounding box feature vector from a feature map. Based on current set-up it must be 8
* *--show* is to  show the results. If it is True opencv will be imported to show the images and bounding boxes
* *--showepi* is to show the epipolar lines at each step. If it is True opencv will be imported to show the images, bounding boxes, and lines
