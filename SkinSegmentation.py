# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:49:40 2019

@author: Rupande Shastri
"""

#SKIN SEGMENTATION FROM FACE
import cv2
import numpy as np
import skfuzzy as fuzz
#function to detect face
path = "classifiers/"
def detect_face(img):
    #loading classifiers
    faceDet = cv2.CascadeClassifier(path+"haarcascade_frontalface_default.xml")
    faceDet2 = cv2.CascadeClassifier(path+"haarcascade_frontalface_alt2.xml")
    faceDet3 = cv2.CascadeClassifier(path+"haarcascade_frontalface_alt.xml")
    faceDet4 = cv2.CascadeClassifier(path+"haarcascade_frontalface_alt_tree.xml")
    #converting color image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    #detecting face using classifiers
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(face) == 1:
        (x,y,w,h) = np.transpose(face)
    elif len(face2) == 1:
        (x,y,w,h) = np.transpose(face2)
    elif len(face3) == 1:
        (x,y,w,h) = np.transpose(face3)
    elif len(face4) == 1:
        (x,y,w,h) = np.transpose(face4)
    else:
        (x,y,w,h) = (0,0,0,0)
        print('error: face not detected/multiple faces detected')
 
    bbox = (x,y,w,h)
    return bbox
#function to crop face from image
def crop_face(img,bbox):
        (x1,y1) = (int(bbox[0] + 0.2*bbox[2]), int(bbox[1]))
        (x2,y2) = (int(bbox[0] + 0.8*bbox[2]), int(bbox[1] + bbox[3]))
        rgb = img[y1:y2, x1:x2]
        try:
            #resizing face for clear visualization
            out = cv2.resize(rgb, (350, 350), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite("cropped.jpg", out) #Write image
            print("file written")
        except:
            print("error")
            pass #If error, pass file
        return out
#skin segmentation
def skin_segment(img):
    img_luv = cv2.cvtColor(img,cv2.COLOR_BGR2Luv)
    init = np.load('init_array2.npy')                            #initial assignment of cluster centres and memberships
    l,u,v = [img_luv[:,:,0],img_luv[:,:,1],img_luv[:,:,2]]      #extracting components
    #calculating distance from centre (vectorized)
    nx,ny = l.shape
    x = np.abs(np.arange(nx) - ((nx - 1)/2))
    y = np.abs(np.arange(ny) - ((ny - 1)/2))
    X,Y = np.meshgrid(x,y)
    dist = np.sqrt(X**2 + Y**2)         #for euclidean distance
    #print(l.shape,u.shape,v.shape,dist.shape)
    mod_data = np.reshape([0.3*l,u,v,0.1*dist],[4,350*350])         #convert to usable form (4 features, 350*350 data)
    c = 2                           #number of cluster centres
    m = 2                           #fuzziness coefficient
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(mod_data, c, m, error=0.005, maxiter=1000, init=init) #clustering
    cluster1 = np.reshape(u[0],[350,350])   #first cluster (due to initial assignment ROI is always in first cluster)
    cluster2 = np.reshape(u[1],[350,350])
    mask = np.zeros([350,350])              #generating mask
    if cluster1[int((nx)/2),int((ny)/2)] > 0.7:
        mask[cluster1 > 0.7] = 255              #membership threshold = 0.7
    else:
        mask[cluster2 > 0.7] = 255    
    return mask

def apply_mask(img,mask):
    mask = mask.astype(np.uint8)
    img = cv2.bitwise_and(img,img,mask=mask)
    return img

#USEAGE
if __name__ == "__main__":
    img = cv2.imread("face2.jpeg")
    out = crop_face(img,detect_face(img))
    mask = skin_segment(out)
    img = apply_mask(out,mask)
    cv2.imshow("Original face",out)
    cv2.waitKey(0)
    cv2.imshow("Mask",img)
    cv2.waitKey(0)