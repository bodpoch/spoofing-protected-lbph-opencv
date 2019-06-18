# -------------------------- TRAINER FOR LBPH ALGORITHM IN FACE RECOGNITION -------------------------------------------
# ---------------------------- BY LAHIRU DINALANKARA AKA SPIKE ----------------------------

import os                                               # importing the OS for path
import cv2                                              # importing the OpenCV library
import numpy as np                                      # importing Numpy library
from PIL import Image                                   # importing Image library

LBPHFace = cv2.face.LBPHFaceRecognizer_create(1, 1, 7,7) # Create LBPH FACE RECOGNIZER

path = 'photodb'                                        # path to the photos
def getImageWithID (path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    FaceList = []
    IDs = []
    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')  # Open image and convert to gray
        faceImage = faceImage.resize((150,150))         
        faceNP = np.array(faceImage, 'uint8')           # convert the image to Numpy array
        ID = int(os.path.split(imagePath)[-1].split('.')[1])    # Retreave the ID of the array
        FaceList.append(faceNP)                         # Append the Numpy Array to the list
        IDs.append(ID)                                  # Append the ID to the IDs list
        cv2.imshow('Training Set', faceNP)              # Show the images in the list
        cv2.waitKey(1)
    return np.array(IDs), FaceList                      # The IDs are converted in to a Numpy array
IDs, FaceList = getImageWithID(path)

# ------------------------------------ TRAING THE RECOGNIZER ----------------------------------------
print('TRAINING......')
LBPHFace.train(FaceList, IDs)
LBPHFace.save('recondata/trainingDataLBPH.xml')
print ('XML FILE SAVED...')

cv2.destroyAllWindows()