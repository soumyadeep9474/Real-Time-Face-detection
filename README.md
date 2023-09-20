# Real-Time-Face-detection
This is a real-time security camera application built on a flask server to stream video from a Raspberry Pi to the internet. Motion detection was implemented using OpenCV to issue alerts through the Twitter API when motion is detected. Face detection was also implemented to detect and save any faces found in the stream. 
# Project Title: Real-Time Security Camera Application

## Description:

Developed a security camera application using a Raspberry Pi and Flask server to stream video over the internet.
Implemented motion detection and face detection algorithms for enhanced security and monitoring.
## Motion Detection Algorithm:

Created a background image by averaging previous frames.
Subtracted the current frame to identify changes in the scene.
Applied thresholding to highlight areas of significant motion.
Utilized erosion and dilation operations for noise reduction.
Detected motion regions using contour detection.
Implemented bounding boxes around detected motion contours.
Planned future enhancement for multiple bounding boxes to track smaller motions.
## Alerting:

Integrated the Twitter API to issue real-time alerts when motion is detected.
Sent alerts via tweets or direct messages to a specified Twitter account.
## Face Detection:

Incorporated OpenCV for face detection within the video stream.
Identified and saved faces found in the stream.
## Skills Demonstrated:

Computer vision and image processing.
Web application development with Flask.
Integration with external APIs (Twitter API).
Raspberry Pi hardware setup and usage.
## Outcome:

Created a functional and scalable security camera system with motion detection, face detection, and alerting capabilities.
Enhanced real-time monitoring and security for the intended application.
