#python drowniness_yawn.py --webcam webcam_index

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
import os


def sound_alarm(path):
    playsound.playsound(path)

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

def mouth_aspect_ratio(shape):
    vertical = (
        dist.euclidean(shape[61], shape[67])
        + dist.euclidean(shape[62], shape[66])
        + dist.euclidean(shape[63], shape[65])
    )
    horizontal = dist.euclidean(shape[60], shape[64])
    return vertical / (2.0 * horizontal)


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
ap.add_argument(
    "-a",
    "--alarm",
    type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Alert.wav"),
    help="path to alarm WAV file",
)
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.22
EYE_CLOSED_SECONDS = 1.0
MOUTH_AR_THRESH = 0.6
YAWN_SECONDS = 0.8
ALARM_COOLDOWN_SECONDS = 5.0
eye_closed_since = None
yawn_since = None
eye_alerted = False
yawn_alerted = False
last_alarm_time = 0.0

print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
#vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    #for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        mouth_ar = mouth_aspect_ratio(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        now = time.monotonic()
        if ear < EYE_AR_THRESH:
            if eye_closed_since is None:
                eye_closed_since = now
            if now - eye_closed_since >= EYE_CLOSED_SECONDS:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not eye_alerted:
                    eye_alerted = True
                    if args["alarm"] and now - last_alarm_time >= ALARM_COOLDOWN_SECONDS:
                        Thread(
                            target=sound_alarm,
                            args=(args["alarm"],),
                            daemon=True,
                        ).start()
                        last_alarm_time = now

        else:
            eye_closed_since = None
            eye_alerted = False

        if mouth_ar > MOUTH_AR_THRESH:
            if yawn_since is None:
                yawn_since = now
            if now - yawn_since >= YAWN_SECONDS:
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not yawn_alerted:
                    yawn_alerted = True
                    if args["alarm"] and now - last_alarm_time >= ALARM_COOLDOWN_SECONDS:
                        Thread(
                            target=sound_alarm,
                            args=(args["alarm"],),
                            daemon=True,
                        ).start()
                        last_alarm_time = now
        else:
            yawn_since = None
            yawn_alerted = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MOUTH: {:.2f}".format(mouth_ar), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if len(rects) == 0:
        eye_closed_since = None
        yawn_since = None
        eye_alerted = False
        yawn_alerted = False

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
