import numpy as np
import argparse
import cv2
import datetime
import imutils
import os
import threading
from computerVision.imageCompression import imageCompressor
from userAlerts.twitterAlert import TwitterCommunicator
import logging

class FaceDetector:
    def __init__(self, prototxt, model):
        logging.debug("Initializing FaceDetector")
        self.prototxt = prototxt
        self.model = model
        self.loadModel()
        self.compressor = imageCompressor.ImageCompressor()
        self.twitterComm = TwitterCommunicator()

    def loadModel(self):
        logging.debug("Loading caffe model")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

    def detectFaces(self, image, sendAlert):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if (confidence > 0.5):
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")
                width = endX - startX
                height = endY - startY

                face = image[startY: endY, startX: endX]
                #face = imutils.resize(face, width = width + 100)
                #Ensure that the extracted face isn't none to defend against image saving errors
                if (face is not None):
                    #Set up multithreading below to use when implementing image compression 
                    #(as image compression is costly) and takes some time
                    #face = self.compressor.svdCompress(face)
                    #saveFace = threading.Thread(target=self.saveImg, args=(face, ))
                    #saveFace.start()
                    logging.info("Face detected. Saving image of face")
                    if (sendAlert):
                        now = datetime.datetime.now()
                        current_time = now.strftime("%H.%M.%S")
                        message = "A face has been detected @" + current_time
                        self.twitterComm.directMessage(message)
                    self.saveImg(face)
                    return True
        
        logging.info("No face detected")
        return False

    def saveImg(self, image):
        #If it doesn't already exist, create a directory for storing the detected faces
        if (not os.path.isdir("faces")):
            logging.info("Faces directory doesn't exist. Creating directory")
            os.mkdir("faces")

        #Construct the fileName for each detected face using a timestamp
        now = datetime.datetime.now()
        current_time = now.strftime("%H.%M.%S")
        fileName = current_time + " - detected" + ".jpg"
        i = 0
        # If the file already exists, we keep looping to define a unique 
        # filename using index as unique identifier

        while (os.path.isfile(os.path.join("faces", fileName))):
            fileName = current_time + " - detected (" + str(i) + ")" + ".jpg"
            i += 1

        filePath = os.path.join("faces", fileName)

        cv2.imwrite(filePath, image)

        logging.info(f"Succesfully saved detected face to faces folder - {filePath}")
