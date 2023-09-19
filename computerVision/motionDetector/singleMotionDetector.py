import numpy as np
import imutils
import cv2
import datetime
from requests_oauthlib import OAuth1
import requests
from userAlerts.twitterAlert import TwitterCommunicator
import logging

class SingleMotionDetector:
    #initialize with accumulative weight and set background to none
    def __init__(self, accumulativeWeight=0.5):
        #Setting accumWeight to 0.5 initially to evenly weigh the initial bg
        logging.debug("Initializing SingleMotionDetector")
        self.accumWeight = accumulativeWeight
        self.bg = None
        self.twitterComm = TwitterCommunicator()

    def update(self, image):
        #Initialize the bg if it hasn't been set yet
        if (self.bg is None):
            self.bg = image.copy().astype("float")
            return
        #Calculate the weighted average
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def detect(self, image, sendAlert, threshVal=25):
        #Calculate the difference between the background and the current image and thresh it
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, threshVal, 255, cv2.THRESH_BINARY)[1]

        #erode and dilate to clean up contours
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        #Grab the contours from the threshed image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        #Initialize (x, y) coords for the corners of the motion
        (lowerX, lowerY) = (np.inf, np.inf)
        (upperX, upperY) = (-np.inf, -np.inf)

        if (len(contours) == 0):
            logging.debug("No motion detected, moving on")
            return None
        
        detectedMotions = []
        #Update the area of motion dependant on the area of the motion contours
        for cont in contours:
            (x, y, w, h) = cv2.boundingRect(cont)
            area = w * h

            if (area > 1000):
                detectedMotions.append((x, y, x + w, y + h))

            # (lowerX, lowerY) = (min(lowerX, x), min(lowerY, y))
            # (upperX, upperY) = (max(upperX, x + w), max(upperY, y + h))
        
        if (sendAlert):
            if (len(detectedMotions) > 0):
                curTime = datetime.datetime.now()
                current_time = curTime.strftime("%H.%M.%S")
                message = "Motion has been detected @" + current_time
                logging.info("Motion detected - Sending motion detection DM")
                self.twitterComm.directMessage(message)
            
        return (thresh, detectedMotions)


def boxIntersect(boxOne, boxTwo):
    logging.debug("Checking box intersection for two detection b-boxes")
    # Given two bounding boxes, determine whether they intersect/overlap
    lowerX1 = boxOne[0]
    lowerY1 = boxOne[1]
    upperX1 = boxOne[2]
    upperY1 = boxOne[3]

    lowerX2 = boxTwo[0]
    lowerY2 = boxTwo[1]
    upperX2 = boxTwo[2]
    upperY2 = boxTwo[3]

    xLow = max(lowerX1, lowerX2)
    xHigh = min(upperX1, upperX2)

    yLow = max(lowerY1, lowerY2)
    yHigh = min(upperY1, upperY2)

    if (xLow > xHigh or yLow > yHigh):
        return False

    return True

def mergeBoxes(boxOne, boxTwo):
    # Given two bounding boxes, merge them into one
    logging.debug("Found overlapping b-boxes. Merging these boxes")
    lowerX = min(boxOne[0], boxTwo[0])
    upperX = max(boxOne[2], boxTwo[2])
    lowerY = min(boxOne[1], boxTwo[1])
    upperY = max(boxOne[3], boxTwo[3])

    return (lowerX, lowerY, upperX, upperY)
