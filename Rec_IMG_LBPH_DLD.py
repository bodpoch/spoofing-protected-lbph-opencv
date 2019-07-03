import cv2  # Importing the opencv
import numpy as np  # Import Numarical Python
import NameFind
import time
import argparse
from imutils.video import VideoStream
import imutils
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args(['-p', 'deploy.prototxt.txt', '-m', 'res10_300x300_ssd_iter_140000.caffemodel']))

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# FACE RECOGNISER OBJECT
LBPH = cv2.face.LBPHFaceRecognizer_create(2, 8, 8, 8, 15)
# Load the training data from the trainer to recognise the faces
LBPH.read("recondata/trainingDataLBPH.xml")

# ------------------------------------  PHOTO INPUT  -----------------------------------------------------

frame = cv2.imread('MYIMG.jpg')  # ------->>> THE ADDRESS TO THE PHOTO

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the Camera to gray
# grab the frame dimensions and convert it to a blob
(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
# pass the blob through the network and obtain the detections and predictions
net.setInput(blob)
detections = net.forward()
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    confidence = detections[0, 0, i, 2]
    if confidence < args["confidence"]:
        continue


    roi_gray = gray[startY:endY, startX:endX]
    # draw the bounding box of the face along with the associated
    # probability
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10

    color = (0, 200, 0)  # BGR 0-255
    stroke = 1

    if (confidence > 0.7):
        cv2.rectangle(gray, (startX, startY), (endX, endY),
                  color, stroke)
        cv2.imwrite('thumb.jpg', roi_gray)
        # time.sleep(5)
        imgtomod = Image.open('thumb.jpg')
        width = imgtomod.size[0]
        height = imgtomod.size[1]
        ideal_width = 150
        ideal_height = 150
        ideal_aspect = ideal_width / float(ideal_height)
        aspect = width / float(height)

        if aspect > ideal_aspect:
            # Then crop the left and right edges:
            new_width = int(ideal_aspect * height)
            offset = int((width - new_width) / 2)
            thumb = imgtomod.crop((offset, 0, width - offset, height)).resize((ideal_width, ideal_height),
                                                                              Image.ANTIALIAS)
            thumb.save('thumb.jpg')
        else:
            # ... crop the top and bottom:
            new_height = int(width / ideal_aspect)
            offset = int((height - new_height) / 2)
            thumb = imgtomod.crop((0, offset, width, (height - offset))).resize((ideal_width, ideal_height),
                                                                                Image.ANTIALIAS)
            thumb.save('thumb.jpg')

        roi_mod = cv2.imread('thumb.jpg')
        roi_mod = cv2.cvtColor(roi_mod, cv2.COLOR_BGR2GRAY)

        # ... precision % of face detection
        # cv2.putText(frame, text, (startX, y),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        ID, conf = LBPH.predict(roi_mod)  # Determine the ID of the photo
        confval = 4
        if conf < confval:
            NAME = NameFind.ID2Name(ID, conf)
            # NameFindForDL.DispID(startX, startY, endX-startX, endY-endY, NAME,gray)
            cv2.putText(gray, NAME, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            print(ID, NAME, '|', "{:.2f}".format(100 - ((conf - 1) / (confval - 1)) * 100), ' %')

            # NameFindForDL.DispID(startX, startY, endX, endY, NAME, gray)                
        else:
            print('Not recognised')
            NAME = NameFind.ID2Name(0, conf)
            cv2.putText(gray, NAME, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow('LBPH Deep Learning Photo Recogniser', gray)  # IMAGE DISPLAY
cv2.waitKey(0)
cv2.destroyAllWindows()