# Multi-view multi-object bounding-box re-identification in a three-camera system installed in a shopping cart

* Dataset: A dataset of images captured by a three-camera system installed inside a smart shopping
* Camera Calibration: Implementing the Tsai method for camera calibration, ensuring accurate geometric calibration of the camera system to correct lens distortion and intrinsic parameters
* Epipolar Geometry Pipeline: Developing an epipolar geometry pipeline based on fundamental and essential matrices
* 3D Localization Estimation: Designing and implementing an approach for 3D localization estimation without computing a depth map, leveraging geometric constraints and feature triangulation techniques
* Siamese Network Training: Modeling and training a Siamese neural network architecture based on triplet loss for bounding box re-identification purposes, enabling robust and discriminative feature embeddings
* End-to-end Multi-View Bounding Box Re-Identification algorithm: Developing a multi-object multi-camera bounding box re-identification algorithm, integrating the trained Siamese network with epipolar constraints to enable accurate and efficient object re-identification across different camera views
* Deployment Automation: Implementing deployment automation solutions using C++ for Edge-TPU Coral and Raspberry Pi platforms, optimizing inference performance and resource utilization for real-time application deployment at the edge.

A sample result from the main  multi-view multi-object bounding-box re-identification algorithm in the presence of noise (inaccurate bounding boxes detected by an object detection model):

![results](https://github.com/shahrokh1106/multiview-boundingbox-reidentification-smart-cart/assets/44213732/bbae13c3-cb97-45a1-afea-bcfb479d0ecd)

An example of using epipolar constraint when the bounding boxes are matched vs when we have unmatched bounding boxes the main  multi-view multi-object bounding-box re-identification algorithm:
![epipolar](https://github.com/shahrokh1106/multiview-boundingbox-reidentification-smart-cart/assets/44213732/e8805a16-3578-4bf1-ba91-25990e6bba99)
