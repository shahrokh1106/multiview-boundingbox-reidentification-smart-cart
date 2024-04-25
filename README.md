# Multi-view bounding-box re-identification in a three-camera system installed in a shopping cart

* Dataset Management: Managed and curated a comprehensive dataset of images captured by a three-camera system installed inside a smart shopping cart and ensured data quality, consistency, and relevance for training and evaluation purposes
* Camera Calibration: Implemented the Tsai method for camera calibration, ensuring accurate geometric calibration of the camera system to correct lens distortion and intrinsic parameters
* Epipolar Geometry Pipeline: Developed an epipolar geometry pipeline based on fundamental and essential matrices
* 3D Localization Estimation: Designed and implemented a novel approach for 3D localization estimation without computing a depth map, leveraging geometric constraints and feature triangulation techniques
* Siamese Network Training: Modeled and trained a Siamese neural network architecture based on triplet loss for bounding box re-identification purposes, enabling robust and discriminative feature embeddings
* End-to-end Multi-View Bounding Box Re-Identification algorithm: Developed a multi-object multi-camera bounding box re-identification algorithm, integrating the trained Siamese network with epipolar constraints to enable accurate and efficient object re-identification across different camera views
* Deployment Automation: Implemented deployment automation solutions using C++ for Edge-TPU Coral and Raspberry Pi platforms, optimizing inference performance and resource utilization for real-time application deployment at the edge.
