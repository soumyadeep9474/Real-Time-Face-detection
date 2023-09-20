# Real-Time-Face-detection
This is a real-time security camera application built on a flask server to stream video from a Raspberry Pi to the internet. Motion detection was implemented using OpenCV to issue alerts through the Twitter API when motion is detected. Face detection was also implemented to detect and save any faces found in the stream. 
Motion Detection Algorithm:

Weighting of Previous Frames: This step involves creating a weighted average of previous frames to form a "background" image. This helps in identifying static elements in the video stream.

Frame Subtraction: The current frame is subtracted from the weighted average to obtain the "difference" image. This is where moving objects or changes in the scene are highlighted.

Thresholding: Thresholding is applied to the difference image to convert it into a binarized image. This process helps distinguish areas with significant differences from the static background.

Noise Reduction: Erosion and dilation operations are used to clean up noise from the thresholded image. This step helps in refining the regions of interest and removing false positives.

Contour Detection: Contour detection is performed on the cleaned image to identify and locate areas of motion detection. Each contour represents a distinct region where motion has been detected.

Bounding Box: A bounding box is drawn around the detected motion contour. Currently, your implementation supports a single bounding box for all movement, but you plan to implement multiple boxes for smaller motions in the future.

Alerting via Twitter API:

When motion is detected, your application uses the Twitter API to send alerts. This could involve sending tweets or direct messages to a specified Twitter account or group. It's a useful way to notify users or take further actions when security events occur.
Face Detection:

You've also implemented face detection as part of your application. This feature involves using OpenCV to identify and locate faces within the video stream.
Future Enhancements:

You mentioned the intention to implement different bounding boxes for smaller motions. This enhancement could provide more precise tracking of multiple moving objects.
