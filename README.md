# Facial recognition system with local binary patterns histogram (LBPH) algorithm and spoofing protection


## APPLICATIONS ARE USING OPENCV 4

## Quick tutorial

To get started you need to to have python 3 set up on your machine as well as the OpenCV library v 4.0.0 or higher:
https://github.com/opencv/opencv/. You may need to install additional
python libraries, such as: numpy, scipy, imutils and dlib.

Initially, database contains 3 persons. You can clear **Names.txt** file and add
some photos to photodb folder using **Face_Capture_With_Rotate.py** or manually.
After that you need to run **Trainer.py** to create XML data for recognition method.


  - Use **Rec_IMG_LBPH.py** or **Rec_IMG_LBPH_DLD.py** to recognise image
  - Use **Rec_LBPH_Antispoofing.py** to 
do real-time recognition using camera or videostream

Both files have parameters *confval* that are used as threshold
and to indicate relative recognition percent rate. Low *confval* means high FRR, high *confval* means high FAR in terms 
of biometric security error. You can manipulate with **cv2.face.LBPHFaceRecognizer_create(radius, neghbors, grid_x, grid_y, threshold)** method parameters 
in order to get better results. Read more info about OpenCV LBPH method here: https://docs.opencv.org/4.1.0/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html

## FILES

**Face_Capture_With_Rotate.py:** Running this file will capture 50 images of a person infront of with illumination and rotation correction.

**NameFind.py:** This file contains all the functions.

**Trainer.py:** This file will train LBPH recognition algorithm using the images in the photodb folder.

**Rec_LBPH_Antispoofing.py:**  This script will recognise faces from the camera feed using LBPH facial recognition algorithm and eye-blink based
liveness detection.

**Detect_blinks.py:** This file will detect blinks in the camera stream

**Rec_IMG_LBPH.py:** Script to recognise faces in the sample image using Haar-cascades and LBPH

**Rec_IMG_LBPH_DLD.py** Script to recognise faces in the sample image using Deep Learning detection and LBPH

**shape_predictor_68_face_landmarks.dat:** pre-trained dlib facial landmark predictor

**res10_300x300_ssd_iter_140000.caffemodel** Caffe framework model for face detection

**MYIMG.jpg:** sample image

**thumb.jpg** subimage

**Names.txt:** contains ID and NAME of all persons in the database
							

## FOLDERS

**photodb** --> Contains the images that will be used to train the recogniser

**Haar** --> Contains Haar Cascades of OpenCV used in the applications

**recondata** --> Contains the saved XML file with LBPH training data

License
----

MIT