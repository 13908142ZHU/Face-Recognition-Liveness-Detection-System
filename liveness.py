"""Machanism: students can finish blinking actively, and photos and videos can't simulate effective blinking.
Through counting blinking times, liveness could be detected.
"""
import numpy as np
import time
from config import EAR_THRESHOLD, BLINKS_REQUIRED, LIVENESS_TIME,EAR_FRAMES, HOLD_TIME
blink_times = []
close_frames = 0
alive_until = 0

"""EAR: a metric that quantifies eye openness by calculating the ratio of the sum of vertical eye landmark distances to 
twice the horizontal eye landmark distance.
When eyes are open, vertical distance is large.
When eyes are closed, vertical distance is small, but horizontal distance keeps almost constant.
the ration could show."""
def EAR(eye_points):
    p1 = eye_points[0]
    p2 = eye_points[1]
    p3 = eye_points[2]
    p4 = eye_points[3]
    p5 = eye_points[4]
    p6 = eye_points[5]

    #Calculate the distance(Euclidean distance)between two points.
    #Get the value of EAR.
    def distance(a,b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    ver_distance = distance(p2, p6) + distance(p3, p5)
    hor_distance= distance(p1, p4)

    ear = ver_distance / (2 * hor_distance)
    return ear

#When EAR is less than a threshold, we think that eyes are closed.
#When EAR is above it, we think that eyes are open, and we need to check how many frames that the period of eyes closing.
def liveness_update(left_eye_points, right_eye_points):

    global close_frames, blink_times, alive_until

    left_ear = EAR(left_eye_points)
    right_ear = EAR(right_eye_points)
    average_ear = (left_ear + right_ear) / 2

    now = time.time()

    print(f"EAR={average_ear:.3f}, close_frames={close_frames}, blinks={len(blink_times)}")

    #If ear < threshold, it could be thought that eyes are closed.
    if average_ear < EAR_THRESHOLD:
        close_frames += 1
    #Else, eyes are open.
    else:
        #If the number of frames that eyes are closed is larger than EAR_FRAMES, it could be recorded as an valid blink.
        if close_frames >= EAR_FRAMES:
            blink_times.append(now)
        close_frames = 0

    #Only retain the number of valid blinks that are within the LIVENESS_TIME.
    new_list = []
    for t in blink_times:
        if now - t <= LIVENESS_TIME:
            new_list.append(t)

    blink_times = new_list

    if len(blink_times) >= BLINKS_REQUIRED:
        alive_until = now + HOLD_TIME

def liveness_detection():
    now = time.time()
    return now <= alive_until
    
def liveness_reset():
    global blink_times, close_frames
    close_frames = 0
    blink_times = []
