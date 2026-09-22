import cv2 as cv
import numpy as np
import face_recognition

from config import VIDEO_SOURCE
from liveness import liveness_update, liveness_detection, liveness_reset
from database import init_db, add_face, match_face
from utils import get_eye_encoding,rescaleFrame

def main():
    init_db()
    liveness_reset()

    capture = cv.VideoCapture(VIDEO_SOURCE)
    mode_register = False
    existing_name = None

    print("BRIEF INTRODUCTION")
    print("Press R to switch register/recognise model.")
    print("Press M to switch to mask model.")
    print("Press S to register current face in register model.")
    print("Press Q to quit.")

    while True:
        #capture is a tuple.
        #The first value is a boolean indicating whether the read operation was successful.
        #The second value is an ndarray representing the captured frame.
        read_result = list(capture.read())
        frame = read_result[1]
        
        if not read_result[0]:
            print("Camera unreadable")
            break

        #Rescale the frame, to imporve processing speed.
        small_frame = rescaleFrame(frame, scale=0.5)

        #Convert rescaled frame from BGR to RGB, Since face_recognition only accepts RGB.
        #Convert rgb_small to uint8, and store continuously.
        rgb_small = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
        rgb_small = np.ascontiguousarray(rgb_small, dtype=np.uint8)

        #Detect location of all faces.
        #Return as list of tuple, e.g. [(top, right, bottom, left),...]
        face_locations = face_recognition.face_locations(rgb_small)

        #Get face encoding(128D) of every face detected.
        face_encodings = face_recognition.face_encodings(rgb_small, known_face_locations=face_locations, num_jitters = 1)

        
        for i in range(len(face_locations)):
            face_pos = face_locations[i]
            face_encoding = face_encodings[i]

            top = face_pos[0]
            right = face_pos[1]
            bottom = face_pos[2]
            left = face_pos[3]

            #Capture faces.
            face_img = rgb_small[top:bottom, left:right]

            #Return a list of dictionaries, and each dictionary contains all the facial landmarks of one face.
            #e.g. [{"left_eye":[...],"right_eye":[...],...},...]
            landmarks_list = face_recognition.face_landmarks(face_img)

            #If no face is detected, skip it and continue the programs.
            if len(landmarks_list) == 0:
                continue
            face_landmarks = landmarks_list[0]

            #Liveness Detection
            liveness_update(face_landmarks["left_eye"], face_landmarks["right_eye"])
            alive = liveness_detection()

            #Green
            if alive:
                color = (0,255,0)
            #Red
            else:
                color = (0,0,255)

            #Draw the boundning box.
            #The coordinates need to be multiplied by 2, because the image is rescaled with 0.5.
            top_big, right_big = top * 2, right * 2
            bottom_big, left_big = bottom * 2, left * 2

            cv.rectangle(frame, (left_big, top_big),(right_big, bottom_big),color, 2)

            #Get eye encoding.
            #If cannot find eye_encoding, use face_encoding.
            eye_encoding = get_eye_encoding(face_img, face_landmarks) 
            if eye_encoding is None:
                eye_encoding = face_encoding

            # Register mode
            if mode_register == True:
                cv.putText(frame, "REGISTER MODE", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.putText(frame, "Press S to save", (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                result = "REGISTERING"
                
            #Recoginize mode
            else:
                cv.putText(frame, "RECOGNISE MODE", (10, 30),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if not alive:
                    result = "NOT ALIVE"
                else:
                    name = match_face(eye_encoding, face_encoding)
                    if name:
                        result = str(name)
                    else:
                        result = "UNKNOWN"

            if result:
                cv.putText(frame, result, (left_big, top_big - 10),cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        display_frame = rescaleFrame(frame, scale=0.5)
        cv.imshow("Face Recognition", display_frame)
        
        #cv2.waitKey(1)will return 32 bits integer.
        #Use cv.waitKey(1) & 0xFF to return 8 bits integer,Which is ASCII code.
        key = cv.waitKey(1) & 0xFF
        if key in(ord("Q"), ord("q")):
            break
        elif key in(ord("r"), ord("R")):
            mode_register = not mode_register
            if mode_register == True:
                print("Switch to REGISTER MODE.")
                liveness_reset()
            else:
                print("Switch to RECOGINISE MODE.")
                liveness_reset()
        elif  key in (ord("s"),ord("S")) and mode_register == True and len(face_encodings) > 0:
            print("Preparing to register...")

            existing_name = match_face(eye_encoding, face_encoding)

            if existing_name is not None:
                print("This face already in the database. Name is:" , existing_name)
            else:
                name = input("Enter name:").strip()
                if name != "":
                    add_face(name,face_encodings[0].astype(np.float32),eye_encoding.astype(np.float32))
                    print("Save successfully.")
                else:
                    print("Name is empty, cannot register.")

    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()