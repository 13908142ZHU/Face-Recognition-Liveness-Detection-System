import numpy as np
import face_recognition
import cv2 as cv

"""
face_img: numpy.ndarray, RGB image of one single face.
face_landmarks: dictionary. It contains 68 coordinates of human faces.
Returns: 
    Success: Return 128-dimensional eye region encoding.
    Fail: Return None.
"""
def get_eye_encoding(face_img, face_landmarks):
    left_eye_points = face_landmarks["left_eye"]
    right_eye_points = face_landmarks["right_eye"]

    eye_points = left_eye_points + right_eye_points

    xs = []
    ys = []

    for p in eye_points:
        x = p[0]
        xs.append(x)

    for p in eye_points:
        y = p[1]
        ys.append(y)

    #Calculate the bounding box. (x_min, y_min) is the top-left corner, and (x_max, y_max) is the bottom-right corner.
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    #Capture eyes region from original face image.
    eye_img = face_img[y_min:y_max, x_min:x_max]

    #Generate list of np.array of eyes region.
    eye_encoding = face_recognition.face_encodings(eye_img)
 
    if len(eye_encoding) == 0:
        return None
    
    # Return the value of eye_encoding. 
    # Because it is a list and a list can't be used, so we need to get the element from it.
    return eye_encoding[0]

def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)