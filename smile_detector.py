import cv2
import numpy as np
import pandas as pd
import sys
from _datetime import datetime

faceCascade = cv2.CascadeClassifier("haarcascade_frontface.xml")
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")

cap = cv2.VideoCapture(0) # Capture the user video and detect the smile of the feature in real time
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

times = []
smile_ratios = []

while True:

    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to gray scale because the opencv detection algorithm works on gray scale images
    # gray = cv2.GaussianBlur(gray, (21, 21), 0) # It should help but it doesn't

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.05,
        minNeighbors = 6,
        minSize = (55, 55),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]

        smile = smileCascade.detectMultiScale(
            gray_face,
            scaleFactor = 1.6,
            minNeighbors = 22,
            minSize = (25, 25),
            flags = cv2.CASCADE_SCALE_IMAGE
            )

        # Set a region of interest for smiles
        for (x, y, w, h) in smile:
            print('Found a smile! (' + str(len(smile))+')')
            cv2.rectangle(color_face, (x, y), (x+w, y+h), (255, 0, 0), 1)

            smile_ratio = str(round(w/x, 3))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Smile meter : ' + smile_ratio, (10, 50), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
            smile_ratios.append(float(smile_ratio))
            times.append(datetime.now())
            print(smile_ratio)

    cv2.imshow('Smile Detector', frame)
    c = cv2.waitKey(7) % 0x100
    if c == 27:
        break

ds={'smile_ratio':smile_ratios,'times':times}
df=pd.DataFrame(ds)
df.to_csv('smile_records.csv')

cap.release()
cv2.destroyAllWindows()
