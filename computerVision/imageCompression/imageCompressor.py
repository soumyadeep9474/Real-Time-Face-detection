import numpy as np
import cv2
import imutils

class ImageCompressor:
    def init(self):
        pass

    def svdCompress(self, image):
        #Load the image and convert it to grayscale
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #use numpy svd to extract the U, S, and VT matrices from the image and operate on the S matrix
        U, S, VT = np.linalg.svd(img, full_matrices=False)
        S = np.diag(S)
        r = 100

        #Generate the compressed image array
        Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r, :]

        return Xapprox


