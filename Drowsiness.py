from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from pygame import mixer  # pip install pygame
from datetime import datetime

mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

def is_nodding(nose_y_hist, nod_thresh=5, min_nods=2):
    if len(nose_y_hist) < 2:
        return False
    nodding = False
    nod_count = 0
    for i in range(1, len(nose_y_hist)):
        if abs(nose_y_hist[i] - nose_y_hist[i - 1]) > nod_thresh:
            nod_count += 1
    if nod_count >= min_nods:
        nodding = True
    return nodding

def is_head_turned(shape):
    # Extract relevant facial landmarks
    left_eye_center = np.mean(shape[36:42], axis=0)
    right_eye_center = np.mean(shape[42:48], axis=0)
    nose_tip = shape[33]

    # Calculate distances from nose tip to eye centers
    left_distance = np.linalg.norm(left_eye_center - nose_tip)
    right_distance = np.linalg.norm(right_eye_center - nose_tip)

    # Define a threshold for what constitutes 'looking sideways'
    sideways_threshold = 1.5  # Adjust based on your observations

    # Check if either eye is significantly further away from the nose tip
    if left_distance > sideways_threshold * right_distance or right_distance > sideways_threshold * left_distance:
        return True
    else:
        return False

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
NOD_THRESH = 5  # Reduced threshold for more sensitivity
MIN_NODS = 5  # Minimum nods detected
NO_BLINK_DURATION = 10  # Duration in seconds to consider no blinking
SIDESWAY_ALERT_DURATION = 2  # Duration in seconds to trigger sideways alert

alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
sideways_timer = None
nose_y_hist = []
open_eyes_start_time = None

print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # Check if any faces are detected
    if len(rects) > 0:
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye[1]
            rightEye = eye[2]

            distance = lip_distance(shape)
            nose_y = shape[33][1]  # Nose tip y-coordinate

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH and not is_head_turned(shape):
                COUNTER += 1
                open_eyes_start_time = None
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not mixer.music.get_busy():
                        mixer.music.play()
            else:
                if open_eyes_start_time is None:
                    open_eyes_start_time = datetime.now()
                elif (datetime.now() - open_eyes_start_time).total_seconds() > NO_BLINK_DURATION and not is_head_turned(shape):
                    cv2.putText(frame, "NO BLINKING ALERT!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not mixer.music.get_busy():
                        mixer.music.play()
                COUNTER = 0

            if distance > YAWN_THRESH:
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():
                    mixer.music.play()
            else:
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Nodding detection
            nose_y_hist.append(nose_y)
            if len(nose_y_hist) > 10:  # Maintain a history of the last 10 y-coordinates
                nose_y_hist.pop(0)

            if is_nodding(nose_y_hist, NOD_THRESH, MIN_NODS):
                cv2.putText(frame, "NODDING ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():
                    mixer.music.play()

            # Check if the driver is looking sideways
            if is_head_turned(shape):
                if sideways_timer is None:
                    sideways_timer = datetime.now()
                elif (datetime.now() - sideways_timer).total_seconds() > SIDESWAY_ALERT_DURATION:
                    cv2.putText(frame, "LOOKING SIDEWAYS FOR TOO LONG!", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not mixer.music.get_busy():
                        mixer.music.play()
            else:
                sideways_timer = None  # Reset timer if not looking sideways
                cv2.putText(frame, "HEAD STRAIGHT", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
