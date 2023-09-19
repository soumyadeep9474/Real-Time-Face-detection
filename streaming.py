from computerVision.motionDetector import singleMotionDetector as smd
from computerVision.faceDetector import faceDetector
import numpy as np
from flask import Flask, render_template, Response, redirect
import threading
import argparse
import datetime
import time
from imutils.video import VideoStream
import cv2
import imutils
import logging

#Const variables used for the face detection model
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_PATH = "deploy.prototxt.txt"

#Init a lock for thread safety when exchanging output frames
outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

#Start the video stream and let it sleep for 2 seconds so the camera can warm up
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def default():
    return redirect('/home')

#Default path routing
@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template("index.html")

def detectMotion(frameCount):
    #Obtain global refs to the videostream, output frame, and lock
    global vs, outputFrame, lock, MODEL_PATH, PROTOTXT_PATH

    md = smd.SingleMotionDetector(accumulativeWeight=0.1)
    fd = faceDetector.FaceDetector(prototxt=PROTOTXT_PATH, model=MODEL_PATH)
    total = 0
    frameNum = 0
    dmCountFace = 0
    dmCountMotion = 0

    while True:
        #Obtain the frame and greyscale and slightly blur it
        frameNum += 1
        # Count the total number of passed frames before sending a Twitter Direct Message
        # To ensure the Twitter API doesn't block us for too many requests in a short timespan
        dmCountMotion += 1
        dmCountFace += 1
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        print(f"dmCountMotion {dmCountMotion}")
        print(f"dmCountFace {dmCountFace}")

        #Obtain the timestamp and add it to the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        if (total > frameCount):
            #If sufficient frames have come by, detect for motion
            # Only send alerts every > 300 frames so not to get blocked by twitter api
            if (dmCountMotion > 300):
                # detect motion with the send alert flag true if greater than 300 frames have passed
                motion = md.detect(gray, sendAlert=True)

                # Only reset the dmCountMotion if we passed the send alert flag as true and didn't detect any motion
                if (motion is not None):
                    if (len(motion[1]) > 0):
                        dmCountMotion = 0
                        logging.info("Sending motion detected DM")
                        logging.info("Resetting motion count counter")
            else:
                motion = md.detect(gray, sendAlert=False)
            
            if (motion is not None):
                # Only check for faces every 20 frames (for performance speed)
                if (frameNum >= 20):
                    # Ensure that we're waiting 300 frames before sending dms
                    if (dmCountFace > 300):

                        ret = fd.detectFaces(image=frame, sendAlert=True)
                        if (ret == True):
                            logging.info("Sending face detected DM")
                            logging.info("Resetting face count counter")
                            dmCountFace = 0
                        
                        frameNum = 0

                    # If it hasn't been 300 frames just detect faces without sending an alert
                    else:
                        fd.detectFaces(image=frame, sendAlert=False)
                        frameNum = 0

                # Extract the detected motions and draw them onto the image
                # TODO: merge overlapping detection boxes into one box
                (thresh, motions) = motion
                for singleMotion in motions:
                    (lowerX, lowerY, upperX, upperY) = singleMotion
                    cv2.rectangle(frame, (lowerX, lowerY), (upperX, upperY), (0, 0, 255), 2)
        
        #Update the background and increment tht total num of frames
        md.update(gray)
        total += 1

        #Ensure the outputFrame is not being read while also being updated
        #and copy it to the output frame
        with lock:
            outputFrame = frame.copy()

def generate():
    #Obtain global refs to the lock and outputFrame
    global outputFrame, lock

    #Loop over the output stream frames
    while True:
        #Acquire the lock so that outputframe is not read while being updated
        with lock:
            #If the outputframe isn't set, continue looping
            if (outputFrame is None):
                continue

            (succesfullyEncoded, encodedImg) = cv2.imencode(".jpg", outputFrame)

            #If the image wasn't succesfully encoded, continue looping
            if (not succesfullyEncoded):
                continue

        #Yield the output frame in a byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImg) + b'\r\n')


@app.route("/videoStream", methods=["GET"])
def videoStream():
    #Return the generated response along with the media type
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if (__name__ == "__main__"):
    # Setup for basic logging
    logging.basicConfig(filename="secureStream.log", level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("Starting stream")

    # Build the argument parser for the cli
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", help="ip address of the device", 
        type=str, required=True)
    ap.add_argument("-o", "--port", help="port of the server (1024 to 65535",
        type=int, required=True)
    ap.add_argument("-f", "--frame_count", help="minimum number of frames required to form background",
        type=int, default=32)
    args = vars(ap.parse_args())

    # Set up the threading for concurrency handling
    t = threading.Thread(target=detectMotion, args=(args["frame_count"],))
    t.daemon = True
    t.start()


    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

vs.stop()