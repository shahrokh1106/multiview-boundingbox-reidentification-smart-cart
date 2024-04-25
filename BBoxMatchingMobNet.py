import numpy as np
import json
import tensorflow as tf
import pickle
import argparse

class BboxMatching():
    """
    The main class for bounding-box matching algorithm
    """
    def __init__(self,
                 ImgHeight = 320,
                 ImgWidth = 320,
                 UseEppipolarFlag = True,
                 UseEmbeddingFlag = True,
                 EmbeddingModelPath = "EmbeddingModelMobNet.tflite",
                 CalibDataPath = ["calib_data_cam0.json","calib_data_cam1.json","calib_data_cam2.json"],
                 BboxScaling = 8,
                 Debug = False,
                 ShowEppipolar = False):
        """
        ImgHeight and ImgWidth are the size of input images which must match the calibration set-up
        UseEppipolarFlag determines if the algorithm should use eppipolar constraints or not 
        UseEmbeddingFlag determines if the algorithm should use the pretrained embedding model or not
        EmbeddingModelPath determines the path of the embedding model trained based on an triplet-loss function
        CalibDataPath is a list of three string paths that indicates the location of json calib files for cam0, cam1, and cam2 respectively
        BboxScaling is an integer that indicates the required scaling when getting a bounding box feature vector from a featur map. Based on current set-up it must be 32
        Debug is to  show the results. If Debug is True opencv will be imported to show the images and bounding boxes
        ShowEppipolar is to show the epipolar lines. If ShowEppipolar is True opencv will be imported to show the images and bounding boxes
        """
        self.UseEmbeddingFlag = UseEmbeddingFlag
        self.UseEppipolarFlag = UseEppipolarFlag 
        self.ImgHeight = ImgHeight
        self.ImgWidth = ImgWidth 
        self.EmbeddingModelPath = EmbeddingModelPath 
        self.EmbeddingInterpreter = self.LoadEmbeddingModel() # loading an Interpreter for the embedding model (the tflight version)
        self.CalibDataPath = CalibDataPath 
        self.BboxScaling = BboxScaling 
        self.Debug = Debug 
        self.ShowEppipolar  = ShowEppipolar


    def LoadEmbeddingModel(self):
            """
            It loads the embedding tflite model based on embedding_model_path
            It returns a tflite interpreter
            """
            interpreter = tf.lite.Interpreter(self.EmbeddingModelPath,experimental_preserve_all_tensors=True)
            interpreter.allocate_tensors()
            return interpreter
    
    def InterpreterEmbeddingPred (self,input):
        """
        InterpreterEmbeddingPred is a function to predict on the given input based on the embedding model.
        The input is a tensor with shape (1,10,10,1792)
        The output shape is a numpy array with shape (1, 256)
        The output is used for comparing the bounding boxes with respect to the feature distance between their corresponding embedding vectors
        """
        # input = tf.cast((input*255),dtype=np.uint8)
        input_details = self.EmbeddingInterpreter.get_input_details()
        # print(input_details)

        scale = input_details[0]['quantization_parameters']['scales'][0]
        zero_point = input_details[0]['quantization_parameters']['zero_points'][0]

        input = input / scale
        input = input + zero_point

        input = tf.cast(input,dtype=np.uint8)

        output_details = self.EmbeddingInterpreter.get_output_details()
        self.EmbeddingInterpreter.set_tensor(input_details[0]['index'], input)
        self.EmbeddingInterpreter.invoke()
        output_data = self.EmbeddingInterpreter.get_tensor(output_details[0]['index'])
        return output_data

    def LoadCalibData(self, CamIndex):
        """
        It loads the calibration data for a camera indicated by CamIndex.
        CamIndex is an integer value that shows the camera index, must be 0 or 1 or 2 since we have three cameras 
        It returns the camera matrix, pose matrix and the inverse of the pose matrix
        """
        with open(self.CalibDataPath[CamIndex], 'r') as file:
            LoadedCalibrationData = json.load(file)
            K = np.asarray(LoadedCalibrationData['K'])
            RT = np.asarray(LoadedCalibrationData['RT'])
            invRT = np.asarray(LoadedCalibrationData['invRT'])
        return K,RT,invRT
    
    def GetFundamentalMatrix(self, CamNameSourceIndex,CamNameTargetIndex):
        """
        It computes the corresponding 3by3 fundamentl matrix for a given source camera with respect to the target camera
        CamNameSourceIndex and CamNameTargetIndex show the integer indices of the cameras for which we want to compute the fundamental matrix
        The direction for the fundamental matrix is from CamNameSourceIndex to CamNameTargetIndex
        """
        def MakeProductMatrix(vector):
            '''
            It computes the product matrix of a given vector a x b = [a']b, where  a = [a1, a2, a3] and a' = [[0, -a3, a2],[a3, 0, -a1],[-a2, a1, 0]]
            '''
            out = np.zeros((3, 3))
            a1, a2, a3 = vector
            out[0][1] = -a3
            out[0][2] = a2
            out[1][0] = a3
            out[1][2] = -a1
            out[2][0] = -a2
            out[2][1] = a1
            return out
        
        K1,RT1,invRT1 =self.LoadCalibData(CamIndex=CamNameSourceIndex)
        K2,RT2,invRT2 =self.LoadCalibData(CamIndex=CamNameTargetIndex)
        cam_1_to_cam_2 = (RT2@invRT1)[:-1,:]
        R_12 = cam_1_to_cam_2[:,:-1]
        T_12 = cam_1_to_cam_2[:,-1].reshape(3,1)
        T_12_product_matrix = MakeProductMatrix(T_12)
        EssentialMatrix = T_12_product_matrix@R_12
        FundamentalMatrix = np.linalg.inv(K2[:,:-1]).T@EssentialMatrix@np.linalg.inv(K1[:,:-1])
        return FundamentalMatrix

    def ComputeEpipolarLine(self, pixel,CamNameSourceIndex, CamNameTargetIndex):
        """
        Given a pixel in CamNameSourceIndex, it computes the corresponding epipolar line in CamNameTargetIndex
        Inputs: pixel which is a pixel coordinate, CamNameSourceIndex and CamNameTargetIndex show the integer indices of the cameras
        Output: line coefficients a, b ,c that indicates a line using line equation ax + by + c = 0
        """
        FundamentalMatrix = self.GetFundamentalMatrix(CamNameSourceIndex,CamNameTargetIndex)
        coeffs = FundamentalMatrix@np.asarray([[pixel[0],pixel[1],1]]).reshape(3,1)
        return coeffs
    
    def GetBoundingBoxFeatureMap(self,FeatureMap, Bbox):
        """
        Bounding box feature extraction from the feature map
        Inputs: FeatureMap which is a numpy array of shape (1,10,10,1792), and Bbox which is a list of bounding box coordinate values as xmin, ymin, xmax, ymax 
        Outputs: The corresponding feature map for the given bounding box, reshaped to (10,10,1792). The output is a tensor
        """
        DownscaledBox = np.round(Bbox[:4]/self.BboxScaling)
        BboxFeatureMap = FeatureMap[:,int(DownscaledBox[1]):int(DownscaledBox[3]), int(DownscaledBox[0]):int(DownscaledBox[2])]
        BboxFeatureMap = tf.image.resize(BboxFeatureMap,(10,10),method = 'bilinear')
        return BboxFeatureMap
    
    def ComputeCosineSimilarity (self, A, B):
        """
        Compute the cosine similarity between two vectors
        """
        return  1-(np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B)))

    def GetCenterBbox(self, Bbox):
        """
        Computes the center of a given bounding box
        """
        x1 = Bbox[0]
        y1 = Bbox[1]
        x2 = Bbox[2]
        y2 = Bbox[3]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x,center_y)
    
    def GetLineDistance (self,x,y,coeff):
        """
        Computes the distance of the pixel (x,y) to the line with coeff data which includes three parameters indicating a line ax +by + c = 0
        """
        a,b,c = coeff
        return abs((a*x)+(b*y)+c)/np.sqrt((a*a)+(b*b))
    
    def MatchingBoundingBoxes(self, Cam0Data, Cam1Data, Cam2Data):
        """
        Reidentification algorithm for matching bounding boxes across the cameras
        
        Inputs: Cam0Data, Cam1Data, Cam2Data which are the required data (in form of dictionaries) from each camera
        Each disctionary has the following keys: 
        'rgb': a numpy array of shape 320*320*3 showing the corresponding RGB Image

        'bbox': a list of bounding box lists. Each bounding box list has values as xmin, ymin, xmax, ymax
        'features': a numpy array of shape (1, 40, 40, 1792), which is the feature map from Object Detection model (MobNet)
        Output: a list of matching bounding-box IDs like ['0_0','1_0','2_0'] 
                As for  *_*, first * is the camera view and second * is the bounding box number in the corresponding input list of bounding boxes
        """
        
        Cam0RGB,Cam1RGB,Cam2RGB = Cam0Data['rgb'],Cam1Data['rgb'],Cam2Data['rgb'] # Getting RGB images for each camera
        # Getting Bounding box coordinates for each camera view. The values are between 0 and 1, we multiply them by 640 to be compatible with the input size of yolo model (640,640)
        Cam0Bboxes = [bbox*320 for bbox in Cam0Data['bbox']]
        Cam1Bboxes = [bbox*320 for bbox in Cam1Data['bbox']]
        Cam2Bboxes = [bbox*320 for bbox in Cam2Data['bbox']]

        # Check if we have at least one bounding box in each camera view.
        if len(Cam0Bboxes)==0 or len(Cam1Bboxes)==0 or len(Cam2Bboxes)==0:
            print("Prpgram stopped: one of the camera views does not have any bounding box.")
            return -1
        
        #Getting feature map of each camera view. Each feature map is a numpy array of shape (1, 20, 20, 256) obtianed from layer 9 of YOLOv5 small version
        Cam0FeatureMap = Cam0Data['features']
        Cam1FeatureMap = Cam1Data['features']
        Cam2FeatureMap = Cam2Data['features']

        BboxDicAll = dict() # is a disctionary to save and allocate IDs to the bounding boxes of the camera views. 
        for index1, view in enumerate([Cam0Bboxes,Cam1Bboxes,Cam2Bboxes]):
            for index2, box in  enumerate(view):
                # In each ID, first index shows the camera number and second index shows the bounding box number in that view. 
                BboxDicAll.update({str(index1)+'_'+str(index2): box} ) 

        #####################################################################################################################################
        # To show all the input bounding boxes bounding boxes
        if self.Debug == True:
            import cv2
            cam0=Cam0RGB.copy()
            cam1=Cam1RGB.copy()
            cam2=Cam2RGB.copy()
            for key in BboxDicAll.keys():
                bbox = BboxDicAll[key]
                if key[0]=='0':
                    cam0 = cv2.rectangle(cam0,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0,0,255),2 )
                elif key[0]=='1':
                    cam1 = cv2.rectangle(cam1,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0,0,255),2 )
                elif key[0]=='2':
                    cam2 = cv2.rectangle(cam2,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0,0,255),2 )
            cv2.namedWindow("Input-Bounding-Boxes", cv2.WINDOW_NORMAL)
            cv2.imshow("Input-Bounding-Boxes", np.hstack((cam0,cam1,cam2)))
            k = cv2.waitKey(-1) & 0xFF
            if k == ord('s'):
                cv2.imwrite("Input-Bounding-Boxes.png", np.hstack((cam0,cam1,cam2)))
                cv2.destroyAllWindows()      
            else:
                cv2.destroyAllWindows()  
        #####################################################################################################################################
                
        # Here we get the corresponding feature maps for each bounding box with respect to its camera view feature map
        # The goal is to get and compute feature maps all at once
        # if UseEmbeddingFlag==True then the embedding features of the bounding boxes will also be saved using the pretrained embedding model
        BboxFeatureMapsAll = dict()
        if self.UseEmbeddingFlag == True:
            BboxEmbeddingFeaturesAll = dict()  

        for key in BboxDicAll:
            if key[0] == '0':
                BboxFeatureMapCam0 = self.GetBoundingBoxFeatureMap(Cam0FeatureMap,BboxDicAll[key])
                BboxFeatureMapsAll.update({key: BboxFeatureMapCam0.numpy().reshape(-1)})
                if self.UseEmbeddingFlag == True:
                    BboxEmbeddingFeaturesAll.update({key: self.InterpreterEmbeddingPred(BboxFeatureMapCam0).reshape(-1)})
                    
            if key[0] == '1':
                BboxFeatureMapCam1 = self.GetBoundingBoxFeatureMap(Cam1FeatureMap,BboxDicAll[key])
                BboxFeatureMapsAll.update({key: BboxFeatureMapCam1.numpy().reshape(-1)})
                if self.UseEmbeddingFlag == True:
                    BboxEmbeddingFeaturesAll.update({key: self.InterpreterEmbeddingPred(BboxFeatureMapCam1).reshape(-1)})

            if key[0] == '2':
                BboxFeatureMapCam2 = self.GetBoundingBoxFeatureMap(Cam2FeatureMap,BboxDicAll[key])
                BboxFeatureMapsAll.update({key: BboxFeatureMapCam2.numpy().reshape(-1)})
                if self.UseEmbeddingFlag == True:
                    BboxEmbeddingFeaturesAll.update({key: self.InterpreterEmbeddingPred(BboxFeatureMapCam2).reshape(-1)})
      
        # Next, we get the IDs of each bounding box in each view to compute all the possible combinations of triple bounding boxes
        Cam0BoxKeys = [key for key in BboxDicAll.keys() if key[0]=='0']
        Cam1BoxKeys = [key for key in BboxDicAll.keys() if key[0]=='1']
        Cam2BoxKeys = [key for key in BboxDicAll.keys() if key[0]=='2'] 
        # Combinations is a list of all possible triple combinations of bounding boxes from all camera
        Combinations = [(bbox0,bbox1,bbox2) for bbox0 in Cam0BoxKeys for bbox1 in Cam1BoxKeys for bbox2 in Cam2BoxKeys]
        
        # Defining a list to save later similarity costs; 
        # This similarity could be from directly yolo feature maps or from embedding model depending on UseEmbeddingFlag value
        SimilarityDistanceCosts = []
        # Defining a list to save th later epipolar costs if UseEppipolarFlag is True
        if self.UseEppipolarFlag==True:
            EpipolarDistanceCosts = []

        for Triple in Combinations: # iterating in all possible triple matches
            Cam0BboxKey,Cam1BboxKey,Cam2BboxKey = Triple
            # Getting the cosine similarity measurements for all the pairs and add them all to get the similarity cost for the corresponding possible triple matching
            if self.UseEmbeddingFlag==True:
                SimCost01 = self.ComputeCosineSimilarity(BboxEmbeddingFeaturesAll[Cam0BboxKey],BboxEmbeddingFeaturesAll[Cam1BboxKey])
                SimCost02 = self.ComputeCosineSimilarity(BboxEmbeddingFeaturesAll[Cam0BboxKey],BboxEmbeddingFeaturesAll[Cam2BboxKey])
                SimCost12 = self.ComputeCosineSimilarity(BboxEmbeddingFeaturesAll[Cam1BboxKey],BboxEmbeddingFeaturesAll[Cam2BboxKey]) 
            else:
                SimCost01 = self.ComputeCosineSimilarity(BboxFeatureMapsAll[Cam0BboxKey],BboxFeatureMapsAll[Cam1BboxKey])
                SimCost02 = self.ComputeCosineSimilarity(BboxFeatureMapsAll[Cam0BboxKey],BboxFeatureMapsAll[Cam2BboxKey])
                SimCost12 = self.ComputeCosineSimilarity(BboxFeatureMapsAll[Cam1BboxKey],BboxFeatureMapsAll[Cam2BboxKey]) 
                
            SimilarityDistanceCosts.append((SimCost01+SimCost02+SimCost12)/3) # Total Similarity cost for a given possible triple match

            if self.UseEppipolarFlag==True:

                # The bounding box coordinates need to be divided by 2 before getting the center as they are based on (640,640) image size but camera calibration set-up is based on (320,320)
                CenterBboxCam0 = self.GetCenterBbox(BboxDicAll[Cam0BboxKey])
                CenterBboxCam1 = self.GetCenterBbox(BboxDicAll[Cam1BboxKey])
                CenterBboxCam2 = self.GetCenterBbox(BboxDicAll[Cam2BboxKey])

                # Given the bbox center in cam0, compute the corresponding line in cam1, and then compute the distance of this line to the center of bbox in cam1
                # Then we also compute the distance in the inverse direction
                Coeff01 = self.ComputeEpipolarLine(CenterBboxCam0,int(Cam0BboxKey[0]), int(Cam1BboxKey[0]))
                DistanceLine01 = self.GetLineDistance(CenterBboxCam1[0],CenterBboxCam1[1],Coeff01)
                Coeff10 = self.ComputeEpipolarLine(CenterBboxCam1,int(Cam1BboxKey[0]), int(Cam0BboxKey[0]))
                DistanceLine10 = self.GetLineDistance(CenterBboxCam0[0],CenterBboxCam0[1],Coeff10)

                # Given the bbox center in cam0, compute the corresponding line in cam2, and then compute the distance of this line to the center of bbox in cam2
                # Then we also compute the distance in the inverse direction
                Coeff02 = self.ComputeEpipolarLine(CenterBboxCam0,int(Cam0BboxKey[0]), int(Cam2BboxKey[0]))
                DistanceLine02 = self.GetLineDistance(CenterBboxCam2[0],CenterBboxCam2[1],Coeff02)
                Coeff20 = self.ComputeEpipolarLine(CenterBboxCam2,int(Cam2BboxKey[0]), int(Cam0BboxKey[0]))
                DistanceLine20 = self.GetLineDistance(CenterBboxCam0[0],CenterBboxCam0[1],Coeff20)

                # Given the bbox center in cam1, compute the corresponding line in cam2, and then compute the distance of this line to the center of bbox in cam2
                # Then we also compute the distance in the inverse direction
                Coeff12 = self.ComputeEpipolarLine(CenterBboxCam1,int(Cam1BboxKey[0]), int(Cam2BboxKey[0]))
                DistanceLine12 = self.GetLineDistance(CenterBboxCam2[0],CenterBboxCam2[1],Coeff12)
                Coeff21 = self.ComputeEpipolarLine(CenterBboxCam2,int(Cam2BboxKey[0]), int(Cam1BboxKey[0]))
                DistanceLine21 = self.GetLineDistance(CenterBboxCam1[0],CenterBboxCam1[1],Coeff21)

                # Getting total eppipolar-constraint cost for the given possible triple match
                EpipolarDistanceCosts.append(((DistanceLine01+DistanceLine10)/2+(DistanceLine02+DistanceLine20)/2+(DistanceLine12+DistanceLine21)/2)[0]/3)

                #####################################################################################################################################
                # To show all the epipolar lines. 
                # Given the center of a bounding box in a specific color, you find its corresponding epipolar lines with the same color in the other views
                if self.ShowEppipolar == True:
                    import cv2
                    img_0 = Cam0RGB.copy()
                    img_1 = Cam1RGB.copy()
                    img_2 = Cam2RGB.copy()
                    b = (255,0,0)
                    g = (0,255,0)
                    r = (0,0,255)
                    # drawing the bounding boxes
                    cv2.rectangle(img_0,(int(BboxDicAll[Cam0BboxKey][0]),int(BboxDicAll[Cam0BboxKey][1])),(int(BboxDicAll[Cam0BboxKey][2]),int(BboxDicAll[Cam0BboxKey][3])), b,1)
                    cv2.rectangle(img_1,(int(BboxDicAll[Cam1BboxKey][0]),int(BboxDicAll[Cam1BboxKey][1])),(int(BboxDicAll[Cam1BboxKey][2]),int(BboxDicAll[Cam1BboxKey][3])), g,1)
                    cv2.rectangle(img_2,(int(BboxDicAll[Cam2BboxKey][0]),int(BboxDicAll[Cam2BboxKey][1])),(int(BboxDicAll[Cam2BboxKey][2]),int(BboxDicAll[Cam2BboxKey][3])), r,1)
                    # drawing the center of bounding boxes
                    img_0 = cv2.circle(img_0, (CenterBboxCam0[0],CenterBboxCam0[1]), 2,b, 2)
                    img_1 = cv2.circle(img_1, (CenterBboxCam1[0],CenterBboxCam1[1]), 2,g, 2)
                    img_2 = cv2.circle(img_2, (CenterBboxCam2[0],CenterBboxCam2[1]), 2,r, 2)
                    # drawing the eppipolar line Coeff01 on camera-1 view
                    x1 = int(0)
                    y1 = int(Coeff01[2]/-Coeff01[1])
                    x2 = int(320+1000)
                    y2 = int((Coeff01[0] * x2 + Coeff01[2]) / -Coeff01[1])
                    cv2.line(img_1, (x1,y1), (x2,y2), color = b, thickness = 1) 
                    # drawing the eppipolar line Coeff10 on camera-0 view
                    x1 = int(0)
                    y1 = int(Coeff10[2]/-Coeff10[1])
                    x2 = int(320+1000)
                    y2 = int((Coeff10[0] * x2 + Coeff10[2]) / -Coeff10[1])
                    cv2.line(img_0, (x1,y1), (x2,y2), color = g, thickness = 1) 
                    # drawing the eppipolar line Coeff02 on camera-2 view
                    x1 = int(0)
                    y1 = int(Coeff02[2]/-Coeff02[1])
                    x2 = int(320+1000)
                    y2 = int((Coeff02[0] * x2 + Coeff02[2]) / -Coeff02[1])
                    cv2.line(img_2, (x1,y1), (x2,y2), color = b, thickness = 1)
                    # drawing the eppipolar line Coeff20 on camera-0 view
                    x1 = int(0)
                    y1 = int(Coeff20[2]/-Coeff20[1])
                    x2 = int(320+1000)
                    y2 = int((Coeff20[0] * x2 + Coeff20[2]) / -Coeff20[1])
                    cv2.line(img_0, (x1,y1), (x2,y2), color = r, thickness = 1)
                    # drawing the eppipolar line Coeff12 on camera-2 view
                    x1 = int(0)
                    y1 = int(Coeff12[2]/-Coeff12[1])
                    x2 = int(320+1000)
                    y2 = int((Coeff12[0] * x2 + Coeff12[2]) / -Coeff12[1])
                    cv2.line(img_2, (x1,y1), (x2,y2), color = g, thickness = 1)
                    # drawing the eppipolar line Coeff21 on camera-1 view
                    x1 = int(0)
                    y1 = int(Coeff21[2]/-Coeff21[1])
                    x2 = int(320+1000)
                    y2 = int((Coeff21[0] * x2 + Coeff21[2]) / -Coeff21[1])
                    cv2.line(img_1, (x1,y1), (x2,y2), color = r, thickness = 1)
                    # showing all the lines
                    cv2.namedWindow('Epipolar-Constraints', cv2.WINDOW_NORMAL)
                    cv2.imshow('Epipolar-Constraints', np.hstack((img_0,img_1,img_2)))
                    k = cv2.waitKey(-1) & 0xFF
                    if k == ord('s'):
                        cv2.imwrite("Epipolar-Constraints.png",  np.hstack((img_0,img_1,img_2)))
                        cv2.destroyAllWindows()      
                    else:
                        cv2.destroyAllWindows()
                    #####################################################################################################################################
        # Get the minimum cost match 
        if self.UseEppipolarFlag==True:
            TotalCost = list(((np.asarray(EpipolarDistanceCosts)/320)+np.asarray(SimilarityDistanceCosts))/2)
        else:
            TotalCost = list(np.asarray(SimilarityDistanceCosts))

        MinimumCostMatchIndex = TotalCost.index(min(TotalCost))
        MinimumCost = min(TotalCost)
        MinimumCostMatch = Combinations[MinimumCostMatchIndex]

        if self.Debug==True:
            import cv2
            cam0=Cam0RGB.copy()
            cam1=Cam1RGB.copy()
            cam2=Cam2RGB.copy()
            for key in BboxDicAll.keys():
                if key in MinimumCostMatch:
                    bbox = BboxDicAll[key]
                    if key[0]=='0':
                        cam0 = cv2.rectangle(cam0,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0,255,0),2 )
                    elif key[0]=='1':
                        cam1 = cv2.rectangle(cam1,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0,255,0),2 )
                    elif key[0]=='2':
                        cam2 = cv2.rectangle(cam2,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0,255,0),2 )
                else:
                    bbox = BboxDicAll[key]
                    if key[0]=='0':
                        cam0 = cv2.rectangle(cam0,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0,0,255),2 )
                    elif key[0]=='1':
                        cam1 = cv2.rectangle(cam1,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0,0,255),2 )
                    elif key[0]=='2':
                        cam2 = cv2.rectangle(cam2,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0,0,255),2 )
            cv2.namedWindow("Matching-Bounding-Boxes", cv2.WINDOW_NORMAL)
            cv2.imshow("Matching-Bounding-Boxes", np.hstack((cam0,cam1,cam2)))
            k = cv2.waitKey(-1) & 0xFF
            if k == ord('s'):
                cv2.imwrite("Matching-Bounding-Boxes.png", np.hstack((cam0,cam1,cam2)))
                cv2.destroyAllWindows()      
            else:
                cv2.destroyAllWindows()
        return MinimumCostMatch,MinimumCost


parser = argparse.ArgumentParser()

parser.add_argument('--height',dest="height", type = int, default = 320,
                    help="height of the input images which must match the calibration setup")

parser.add_argument('--width',dest="width", type = int, default = 320,
                    help="width of the input images which must match the calibration setup")

parser.add_argument('--eppipolar',dest="epi", type = bool, default = True,
                    help="determines if the algorithm should use eppipolar constraints or not ")

parser.add_argument('--embedding',dest="embed", type = bool, default = True,
                    help="determines if the algorithm should use the pretrained embedding model or not")

parser.add_argument('--embedpath',dest="embedpath", type = str, default = "EmbeddingModel_MobNet_FeatureBased1.tflite",
                    help="determines the path of the embedding model trained based on an triplet-loss function")

parser.add_argument('--calibpath',dest="calibpath", type = list, default = ["calib_data_cam0.json","calib_data_cam1.json","calib_data_cam2.json"],
                    help="is a list of three string paths that indicates the location of json calib files for cam0, cam1, and cam2 respectively")

parser.add_argument('--boxscaling',dest="boxscaling", type = int, default = 8,
                    help="is an integer that indicates the required scaling when getting a bounding box feature vector from a featur map. Based on current set-up it must be 32")

parser.add_argument('--show',dest="show", type = bool, default = False,
                    help="is to  show the results. If it is True opencv will be imported to show the images and bounding boxes")

parser.add_argument('--showepi',dest="showepi", type = bool, default = False,
                    help="is to show the epipolar lines at each step. If it is True opencv will be imported to show the images, bounding boxes, and lines")

args = parser.parse_args()

def main(args):
    with open('NewSample.pkl', 'rb') as f:
        Data = pickle.load(f)
    Cam0Data = Data[0] # Camera 0 data
    Cam1Data = Data[1] # Camera 1 data
    Cam2Data = Data[2] # Camera 2 data
    MatchingAlg = BboxMatching(UseEppipolarFlag=args.epi,
                               UseEmbeddingFlag=args.embed,
                               Debug = args.show,
                               ImgHeight = args.height,
                               ImgWidth = args.width,
                               EmbeddingModelPath = args.embedpath,
                               CalibDataPath = args.calibpath, 
                               BboxScaling = args.boxscaling,
                               ShowEppipolar=args.showepi)
    MatchedBboxes, Cost = MatchingAlg.MatchingBoundingBoxes(Cam0Data, Cam1Data, Cam2Data)
    print("Matching Bounding-box IDs: ", MatchedBboxes)
    print("Matching Bounding-box Cost: ", Cost)

if __name__ == "__main__":
    main(args)