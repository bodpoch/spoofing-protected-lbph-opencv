import cv2  # Importing the opencv
import numpy as np  # Import Numarical Python
import NameFind
import winsound
import time
import argparse
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import argparse
import dlib


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# import the Haar cascades for face and eye ditection
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

recognise = cv2.face.LBPHFaceRecognizer_create(1, 1, 8, 8, 8)  # LBPH Face recognizer object
recognise.read("recondata/trainingDataLBPH.xml")  # Load the training data from the trainer to recognise the faces

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 1

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
cap = cv2.VideoCapture(0)
vs = cap
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
#time.sleep(1.0)

while True:
    ret, img = cap.read()  # Read the camera object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the Camera to gray
    if (TOTAL > 2):
        TOTAL = 0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect the faces and store the positions


    for (x, y, w, h) in faces:  # Frames  LOCATION X, Y  WIDTH, HEIGHT

        # detect faces in the grayscale frame
        rects = detector(gray, 0)


        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            #cv2.drawContours(gray, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(gray, [rightEyeHull], -1, (0, 255, 0), 1)


            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                # reset the eye frame counter
                COUNTER = 0

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(gray, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(gray, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            if TOTAL <= 0:
                cv2.rectangle(gray, (50, 70), (550, 120), (0,0,0), 2)  # Draw a Black Rectangle over the face frame
                cv2.putText(gray, "Please blink for a security reasons", (100, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            if TOTAL > 0:
                #time.sleep(0.5)
                # The Face is isolated and cropped
                gray_face = gray[y: y + h, x: x + w]
                cv2.imwrite("DLFD/image.jpg", gray_face)
                eyes = eye_cascade.detectMultiScale(gray_face)
                for (ex, ey, ew, eh) in eyes:
                    ID, conf = recognise.predict(gray_face)  # Determine the ID of the photo
                    if conf < 3:
                        NAME = NameFind.ID2Name(ID, conf)
                        NameFind.DispID(x, y, w, h, NAME, gray)
                        print('ID', ID, '-', NAME, "{:.3f}".format(conf), '/',
                              "{:.2f}".format(100 - ((conf - 1) / (3 - 1)) * 100),
                              '%')
                        # print(ID)
                        winsound.PlaySound("audioassist/untitled.wav", winsound.SND_ASYNC | winsound.SND_ALIAS)
                        # time.sleep(0.2)
                        TOTAL = 0
                    else:
                        print('NOT RECOGNIZED')
                        NAME = NameFind.ID2Name(0, conf)
                        NameFind.DispID(x, y, w, h, NAME, gray)

    cv2.imshow('LBPH Face Recognition System with Spoofing-protection', gray)  # Show the video

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit if the key is Q
        break

cap.release()
cv2.destroyAllWindows()