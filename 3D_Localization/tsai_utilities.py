import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2
from math import *
from itertools import combinations
import os
import json


def show (img, scale_percent = 100, waitKey=-1):
    w = int(img.shape[1] * scale_percent / 100)
    h = int(img.shape[0] * scale_percent / 100)
    image = img.copy()
    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    k = cv2.waitKey(waitKey) & 0xFF
    if k == ord('s'):
        cv2.imwrite("image.png", image)
        cv2.destroyAllWindows()     
    if k == ord('q'):
        cv2.destroyAllWindows()  
    cv2.destroyAllWindows()

    
# Tsai Calibration
def homogeneous(vector):
    return np.append(vector, 1)

def translation (point, tx=0, ty=0, tz=0):
    point=homogeneous(point)
    if len(point)==3: 
        translation_matrix = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
    elif len(point)==4: 
        translation_matrix = np.array([[1,0,0,tx],[0,1,ty,0],[0,0,tz,0],[0,0,0,1]])
    new_point = np.matmul(translation_matrix,point)
    return np.delete(new_point,-1)

def Rot2D (point, theta=0):
    point=homogeneous(point)
    rotation_matrix = np.asarray([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0], [0,0,1]])
    new_point = np.matmul(rotation_matrix,point)
    return np.delete(new_point,-1)

def Rot3d(point, thX=0, thY=0, thZ=0):
    point=homogeneous(point)
    rotation_matrix_X = np.asarray([[1,0,0],[0, np.cos(thX), np.sin(thX)], [0, -np.sin(thX), np.cos(thX)]])
    rotation_matrix_Y = np.asarray([[np.cos(thY), 0, -np.sin(thY)],[0,1,0], [np.sin(thY),0, np.cos(thY)]])
    rotation_matrix_Z = np.asarray([[np.cos(thZ), np.sin(thZ), 0],[-np.sin(thZ), np.cos(thZ), 0], [0,0,1]])
    r_XYZ = np.matmul(rotation_matrix_X,rotation_matrix_Y,rotation_matrix_Z)
    rotation_matrix=np.c_[r_XYZ, np.zeros(3)]  
    rotation_matrix=np.r_[rotation_matrix, np.asarray([0,0,0,1]).reshape(1,4)]  
    new_point = np.matmul(rotation_matrix,point)
    return np.delete(new_point,-1)

def scale(point, sx=1, sy=1, sz=1):
    point=homogeneous(point)
    if len(point)==3: 
        scaling_matrix = np.array([[sx,0,0],[0,sy,0],[0,0,1]])
    elif len(point)==4: 
        scaling_matrix = np.array([[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]])
    else: 
        print("This function only works for 2D and 3D points")
    new_point = np.matmul(scaling_matrix,point)
    return np.delete(new_point,-1)

def wrf2crf (point, rot, tvec):
    if tvec.shape!=(3,1):
        tvec.reshape(3,1)
    point=homogeneous(point)
    camera_matrix =np.c_[rot, tvec]
    camera_matrix = np.r_[camera_matrix, np.asarray([0,0,0,1]).reshape(1,4)]
    new_point = np.matmul(camera_matrix,point)
    return np.delete(new_point,-1)

def crf2imp(point, f):
    point=homogeneous(point)
    projection_matrix = np.asarray([[f,0,0,0],[0,f,0,0],[0,0,1,0]])
    new_point = np.matmul(projection_matrix,point)
    temp=new_point[2]
    return np.delete(new_point,-1)/temp

def imp2pix (point, dx, dy, width, height):
    matrix = np.asarray([[1/dx, 0, width/2], [0, -1/dy, height/2], [0,0,1]])
    point=homogeneous(point)
    pixel = np.matmul(matrix,point)
    return np.delete(pixel,-1)

def rms (a,b):
    return sqrt(np.square(np.subtract(a,b)).mean())


def get_homographies(tsaiL,tsaiR):
    M1 = tsaiL.RT
    M2 = tsaiR.RT
    K1 = tsaiL.K
    K2 = tsaiR.K
    P1 = np.dot(K1,M1)
    P2 = np.dot(K2,M2)
    Knew = (K1+K2)/2
    c1 = np.dot(-np.transpose(tsaiL.rot),tsaiL.tvec)
    c2 = np.dot(-np.transpose(tsaiR.rot),tsaiR.tvec)
    baseline = (c1-c2) 
    Vx=baseline.reshape(1,3)
    Vy = np.cross(tsaiL.rot[2],Vx).reshape(1,3)
    Vz = np.cross(Vx, Vy).reshape(1,3)
    rot_new = np.r_[Vx/np.sqrt(np.dot(Vx,Vx.T)),  Vy/np.sqrt(np.dot(Vy,Vy.T)) ,Vz/np.sqrt(np.dot(Vz,Vz.T))]
    M1new = np.r_[np.c_[rot_new, tsaiL.tvec], np.asarray([0,0,0,1]).reshape(1,4)]
    M2new = np.r_[np.c_[rot_new, tsaiR.tvec], np.asarray([0,0,0,1]).reshape(1,4)]
    P1new = np.dot(Knew,M1new)
    P2new = np.dot(Knew,M2new)
    H1 = np.dot(P1new[0:3,0:3],np.linalg.inv(P1[0:3,0:3])) 
    H2 = np.dot(P2new[0:3,0:3],np.linalg.inv(P2[0:3,0:3])) 
    return H1, H2

def make_product_matrix(vector):
    '''
    computes the product matrix of a given vector
    a x b = [a']b,
    where 
    a = [a1, a2, a3] and
    a' = [[0, -a3, a2],
         [a3, 0, -a1],
        [-a2, a1, 0]]
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

def save_calib_data(tsai, camera_name):
    calibration_data = {'K': tsai.K.tolist(),'RT': tsai.RT.tolist(), 'invRT': tsai.invRT.tolist()}
    with open('calib_data_'+camera_name+'.json', 'w') as file:
        json.dump(calibration_data, file)

def load_calib_data(camera_name):
    with open('calib_data_'+camera_name+'.json', 'r') as file:
        loaded_calibration_data = json.load(file)
    K = np.asarray(loaded_calibration_data['K'])
    RT = np.asarray(loaded_calibration_data['RT'])
    invRT = np.asarray(loaded_calibration_data['invRT'])
    return K,RT,invRT

def get_fundamental_matrix(camera_name_1,camera_name_2):
    K1,RT1,invRT1 =load_calib_data(camera_name=camera_name_1)
    K2,RT2,invRT2 =load_calib_data(camera_name=camera_name_2)
    cam_1_to_cam_2 = (RT2@invRT1)[:-1,:]
    R_12 = cam_1_to_cam_2[:,:-1]
    T_12 = cam_1_to_cam_2[:,-1].reshape(3,1)
    T_12_product_matrix = make_product_matrix(T_12)
    essential_matrix = T_12_product_matrix@R_12
    fundamental_matrix = np.linalg.inv(K2[:,:-1]).T@essential_matrix@np.linalg.inv(K1[:,:-1])
    return fundamental_matrix

def compute_epipolar_line(pixel,camera_name_1, camera_name_2):
    # Given a pixel in camera_name_1, compute the corresponding epipolar_line in camera_name_2
    fundamental_matrix = get_fundamental_matrix(camera_name_1,camera_name_2)
    coeffs = fundamental_matrix@np.asarray([[pixel[0],pixel[1],1]]).reshape(3,1)
    return coeffs


class TsaiClass:
    
    def __init__(self, image,pixelsize,points):
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.dx = pixelsize[0]
        self.dy = pixelsize[1]
        self.n=points.shape[0]
        self.realpoints = points[:,0:3]
        self.pixels = points[:,3:5]
        self.impixels = np.c_[self.dx*(self.pixels[:,0]-(self.width/2)),self.dy*(self.height/2-(self.pixels[:,1]))]
        self.L = self.compute_L()
        self.ty, self.sx = self.compute_tysx()
        self.rot, self.tx = self.compute_rottx()
        self.r1 = self.rot[:,0]
        self.r2 = self.rot[:,1]
        self.r3 = self.rot[:,2]
        self.f , self.tz = self.compute_f_tz ()
        self.tvec = np.asarray([self.tx, self.ty, self.tz]).reshape(3,1)
        self.prpixels = self.project_all()
        self.projMatrix = self.get_projectionMatrix ()
        self.RT = np.r_[np.c_[self.rot, self.tvec], np.asarray([0,0,0,1]).reshape(1,4)]
        self.invRT = np.r_[np.c_[(self.rot.transpose(),np.dot(-self.rot.transpose(),self.tvec))],np.asarray([0,0,0,1]).reshape((1,4))]
        self.K = np.dot( np.asarray([[self.sx[0]/self.dx, 0, self.width/2], [0,-1/self.dy,self.height/2], [0,0,1]]), np.asarray([[self.f[0],0,0,0],[0,self.f[0],0,0],[0,0,1,0]]))
        self.bckprpoints = self.get_BackProjection_points()
        self.calibError = rms (self.pixels, self.prpixels)
        self.cubeError = rms (self.realpoints, self.bckprpoints)
        self.kappa = 0
        self.undistort_prpixels = self.undistort_all()
        
    def compute_L(self):
        X=self.realpoints.T[0].reshape(self.n,1)
        Y=self.realpoints.T[1].reshape(self.n,1)
        Z=self.realpoints.T[2].reshape(self.n,1)
        x=self.impixels.T[0].reshape(self.n,1)
        y=self.impixels.T[1].reshape(self.n,1)
        M = np.c_[np.multiply(y,X),np.multiply(y,Y),np.multiply(y,Z),y,np.multiply(-x,X),np.multiply(-x,Y),np.multiply(-x,Z)]
        invM = np.linalg.pinv(M)
        return np.dot(invM, x)
    
    
    
    def compute_tysx (self):
        def distance (a):
            return np.sqrt((a[0]*a[0]) + (a[1]*a[1]))
        def matchSign (a,b):
            return (a < 0.0 and b < 0.0) or (a >= 0.0 and b >= 0.0)
    
        ty = 1/np.sqrt((self.L[4]*self.L[4]) + (self.L[5]*self.L[5]) + (self.L[6]*self.L[6]))
        sx=np.abs(ty)*np.sqrt(self.L[0]**2+self.L[1]**2+self.L[2]**2)
        tx = self.L[3] * ty
        MaxDistanceIndex = np.argmax(np.asarray(list(map(distance,self.impixels))))
        r11=self.L[0]*ty
        r12=self.L[1]*ty
        r13=self.L[2]*ty
        r21=self.L[4]*ty
        r22=self.L[5]*ty
        r23=self.L[6]*ty
        newx=r11*self.realpoints[MaxDistanceIndex][0]+r12*self.realpoints[MaxDistanceIndex][1]+r13*self.realpoints[MaxDistanceIndex][2]+tx
        newy=r21*self.realpoints[MaxDistanceIndex][0]+r22*self.realpoints[MaxDistanceIndex][1]+r23*self.realpoints[MaxDistanceIndex][2]+ty
        if matchSign (newx, self.impixels[MaxDistanceIndex][0]) and matchSign(newy, self.impixels[MaxDistanceIndex][1]):
            return ty,sx
        else:
            return -ty,sx    
        
        
    def compute_rottx(self):
        
        def get_ortho_rot (rot):
            Heading = atan2(-rot[2][0], rot[0][0])
            Attitude = asin(rot[1][0])
            Bank = atan2(-rot[1][2], rot[1][1])
            sa, ca = sin(Attitude), cos(Attitude)
            sb, cb = sin(Bank), cos(Bank)
            sh, ch = sin(Heading), cos(Heading)
            return np.asarray([[ch*ca , (-ch*sa*cb) + (sh*sb) , (ch*sa*sb) + (sh*cb)],
                               [sa , ca*cb , -ca*sb],
                               [-sh*ca, (sh*sa*cb) + (ch*sb), (-sh*sa*sb) + (ch*cb)]])
        
        
        r11 = self.L[0]*(self.ty/self.sx)
        r12 = self.L[1]*(self.ty/self.sx)
        r13 =  self.L[2]*(self.ty/self.sx)
        r21 = self.L[4]*self.ty
        r22 = self.L[5]*self.ty
        r23 = self.L[6]*self.ty
        tx= self.L[3]*(self.ty/self.sx)
        temp = np.cross(np.array([r11,r12,r13]).T,np.array([r21,r22,r23]).T)
        r31  =  temp[0][0]
        r32  =  temp[0][1]
        r33  =  temp[0][2]
        rot = np.array([[r11[0],r12[0],r13[0]],[r21[0],r22[0],r23[0]], [r31,r32,r33]])
        orthonormal_rot = get_ortho_rot (rot)
        return orthonormal_rot,tx
    
    def compute_f_tz(self):
        X=self.realpoints.T[0].reshape(self.n,1)
        Y=self.realpoints.T[1].reshape(self.n,1)
        Z=self.realpoints.T[2].reshape(self.n,1)
        y=self.impixels.T[1].reshape(self.n,1)
        Uy = self.rot[1][0]*X+self.rot[1][1]*Y+self.rot[1][2]*Z+self.ty
        Uz = self.rot[2][0]*X+self.rot[2][1]*Y+self.rot[2][2]*Z
        # Ax=b
        A=np.c_[Uy, -y]
        b= np.multiply(Uz,y)
        invA = np.linalg.pinv(A)
        # return np.dot(invA, b)[0],np.dot(invA, b)[1] 
        return np.dot(invA, b)[0],np.dot(invA, b)[1]
    
    
    def get_projectionMatrix (self):
        camera_matrix =np.c_[self.rot, self.tvec]
        camera_matrix = np.r_[camera_matrix, np.asarray([0,0,0,1]).reshape(1,4)]
        proj_matrix_1 = np.asarray([[self.f[0],0,0,0],[0,self.f[0],0,0],[0,0,1,0]])
        proj_matrix_2 = np.asarray([[1/self.dx, 0, self.width/2], [0, -1/self.dy, self.height/2], [0,0,1]])
        return np.dot(proj_matrix_2,np.dot(proj_matrix_1,camera_matrix))
    

               
    def project_all (self):
        def singleProjection (point):
            crf_point = wrf2crf(point, self.rot, self.tvec)
            imp_point = crf2imp(crf_point, self.f[0])
            prpixel = imp2pix(imp_point, self.dx, self.dy, self.width, self.height)
            return prpixel
        return np.asarray(list(map(singleProjection, self.realpoints)))
    
    
    def undistort_all(self):
        
        def single_undistort (pixel):
            xu = (pixel[0]*(1+self.kappa*(pixel[0]**2+pixel[1]**2)))
            yu = (pixel[1]*(1+self.kappa*(pixel[0]**2+pixel[1]**2)))
            return np.asarray([xu,yu]).reshape(1,2)
        
        def singleProjection_undist (point):
            crf_point = wrf2crf(point, self.rot, self.tvec)
            imp_point = crf2imp(crf_point, self.f[0])
            undist_imp_point = single_undistort(imp_point)
            undist_pixel = imp2pix(undist_imp_point, self.dx, self.dy, self.width, self.height)
            return undist_pixel
        
        undist_pixels = np.asarray(list(map(singleProjection_undist, self.realpoints)))
        return undist_pixels
        

        
    def get_BackProjection_points(self):
        
        def add_f(point):
            return np.append(point, self.f)
        
        def multiply_inverseRT (point):
            inv_RT=np.r_[np.c_[(self.rot.transpose(),np.dot(-self.rot.transpose(),self.tvec))],np.asarray([0,0,0,1]).reshape((1,4))]
            return np.dot(inv_RT,point.T)
        
        
        # inv_RT = np.linalg.pinv(tsai.RT)
        inv_RT=np.r_[np.c_[(self.rot.transpose(),np.dot(-self.rot.transpose(),self.tvec))],np.asarray([0,0,0,1]).reshape((1,4))]
        Oc_world = multiply_inverseRT(np.transpose([0,0,0,1]))
        impixels = self.impixels
        crf_coordinate = np.asarray(list(map (add_f, impixels)))
        crf_coordinate = np.asarray(list(map(homogeneous,crf_coordinate)))
        cube_coordinate = np.asarray(list(map(multiply_inverseRT,crf_coordinate)))
        XPlane = []
        YPlane = []
        for i in range (self.n):
            # x=0
            tx = -Oc_world[0] / (cube_coordinate[i][0] - Oc_world[0])
            # y = 0
            ty = -Oc_world[1] / (cube_coordinate[i][1] - Oc_world[1])
            
            x = Oc_world[0] + tx * (cube_coordinate[i][0] - Oc_world[0])
            y = Oc_world[1] + tx * (cube_coordinate[i][1] - Oc_world[1])
            z = Oc_world[2] + tx * (cube_coordinate[i][2] - Oc_world[2])
            
            xx = Oc_world[0] + ty * (cube_coordinate[i][0] - Oc_world[0])
            yy = Oc_world[1] + ty * (cube_coordinate[i][1] - Oc_world[1])
            zz = Oc_world[2] + ty * (cube_coordinate[i][2] - Oc_world[2])
            XPlane.append([x, y, z])
            YPlane.append([xx, yy, zz])
            
        return np.asarray(XPlane[:len(XPlane) // 2] + YPlane[len(YPlane) // 2:])
    
    def showProjection(self,image,alpha):
       
        fig, (ax) = plt.subplots(1)
        fig.set_size_inches(10,10)
        ax.imshow(image, alpha=alpha)    
        ax.plot(self.pixels[:,0], self.pixels[:,1], 'o', markersize=5, label='real_pixels', color = "b")
        ax.plot(self.prpixels[:,0], self.prpixels[:,1], 'o', markersize=5, label='projected_pixels', color="r")
        ax.set_title('%s' % "results")
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')
        ax.legend()
        fig.savefig("projection.png")
        
    def showBackProjection(self):
        fig = plt.figure()
        fig.set_size_inches(10,10)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.realpoints[:,0], self.realpoints[:,1], self.realpoints[:,2], c='blue', linewidth=0.5, label="Real World Coordinates")
        ax.scatter(self.bckprpoints[:,0], self.bckprpoints[:,1], self.bckprpoints[:,2], c="red", linewidth=0.5, label="Back-projected Coordinates")
        ax.set_title('%s' % "Back-projection results")
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.legend()
        fig.savefig("back_projection.png")

def get_points(img, pattern_size,square_size,left_offset,right_offset, save_flag=True, name="cam_"):
    # Getting corners 3d points and 2d corners
    def click_event(event,x,y,flags,params):
        global points
        if event==cv2.EVENT_LBUTTONDBLCLK:
            points.append([x,y])
            cv2.circle(img,(x,y),1,(0,0,255),-1)
            cv2.imshow("corner_selection", img)
    global points
    points = []
    cv2.namedWindow("corner_selection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("corner_selection",click_event)
    while True:
        cv2.imshow("corner_selection", img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
    points = np.asarray(points)
    
    height = pattern_size[0]
    width = pattern_size[1]
    objp = np.zeros((height*width*2,5),np.float32)
    if len(points) != 0:
        objp[:,3:5] = points.copy()
        count = 0
        for column in range(width): 
            for row in range(height):
                objp[count,1] = column*square_size+left_offset 
                objp[count,2] = row*square_size 
                count+=1
        for column in range(width): 
            for row in range(height): 
                objp[count,0] = column*square_size+right_offset 
                objp[count,2] = row*square_size 
                count+=1
                
        if save_flag:
            np.save("points3d2d_"+name+".npy", objp)
    else:
        print("Not enough corners detected")
        return
    return objp


def get_points_2(img, pattern_size,square_size,left_offset,right_offset, save_flag=True, name="cam_"):
    # Getting corners 3d points and 2d corners
    def click_event(event,x,y,flags,params):
        global points
        if event==cv2.EVENT_LBUTTONDBLCLK:
            points.append([x,y])
            cv2.circle(img,(x,y),1,(0,0,255),-1)
            cv2.imshow("corner_selection", img)
    global points
    points = []
    cv2.namedWindow("corner_selection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("corner_selection",click_event)
    while True:
        cv2.imshow("corner_selection", img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
    points = np.asarray(points)
    
    height = pattern_size[0]
    width = pattern_size[1]
    objp = np.zeros((height*width*2,5),np.float32)
    if len(points) != 0:
        objp[:,3:5] = points.copy()
        count = 0
        for column in range(width): 
            for row in range(height):
                objp[count,1] = -(column*square_size+left_offset) 
                objp[count,2] = (row*square_size) 
                count+=1
        for column in range(width): 
            for row in range(height): 
                objp[count,0] = -(column*square_size+right_offset)
                objp[count,2] = (row*square_size) 
                count+=1
                
        if save_flag:
            np.save("points3d2d_"+name+".npy", objp)
    else:
        print("Not enough corners detected")
        return
    return objp


def tsai_calibration(image,pattern_size,square_size,left_offset,right_offset,pixel_size,save_flag, camera_name, debug=False):
    img = image.copy()
    if os.path.exists("points3d2d_"+camera_name+".npy"):
        points3d2d = np.load("points3d2d_"+camera_name+".npy")
    else: 
        points3d2d =  get_points(img, pattern_size,square_size,left_offset,right_offset, save_flag, camera_name)
    tsai = TsaiClass(image, pixel_size, points3d2d)
    save_calib_data(tsai, camera_name)
    if debug:
        tsai.showProjection(image,alpha=0.7)
        tsai.showBackProjection()
    return tsai

def tsai_calibration_2(image,pattern_size,square_size,left_offset,right_offset,pixel_size,save_flag, camera_name, debug=False):
    img = image.copy()
    if os.path.exists("points3d2d_"+camera_name+".npy"):
        points3d2d = np.load("points3d2d_"+camera_name+".npy")
    else: 
        points3d2d =  get_points_2(img, pattern_size,square_size,left_offset,right_offset, save_flag, camera_name)
    tsai = TsaiClass(image, pixel_size, points3d2d)
    save_calib_data(tsai, camera_name)
    if debug:
        tsai.showProjection(image,alpha=0.7)
        tsai.showBackProjection()
    return tsai
