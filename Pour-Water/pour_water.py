import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

cv2.namedWindow("Pour Water", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Pour Water", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

water_level = 0

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    angle = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            wrist = hand_landmarks.landmark[0]
            mid = hand_landmarks.landmark[9]

            x1 = int(wrist.x*w)
            y1 = int(wrist.y*h)

            x2 = int(mid.x*w)
            y2 = int(mid.y*h)

            angle = math.degrees(math.atan2(y2-y1,x2-x1))

            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),3)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # container
    box_x = w//2 + 200
    box_y = h//2 + 150

    cv2.rectangle(frame,
                  (box_x-50,box_y-200),
                  (box_x+50,box_y),
                  (255,255,255),
                  3)

    # water fill
    cv2.rectangle(frame,
                  (box_x-50,box_y-water_level),
                  (box_x+50,box_y),
                  (255,0,0),
                  -1)

    # pouring logic
    if -140 < angle < -90:

        water_level += 3

    elif angle < -160:

        cv2.putText(frame,"SPILL!",
                    (w//2-100,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0,0,255),
                    3)

    # limit water
    water_level = min(water_level,200)

    cv2.putText(frame,f"Angle: {int(angle)}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2)

    # exit button
    button_x1 = w - 180
    button_y1 = 20
    button_x2 = w - 20
    button_y2 = 80

    cv2.rectangle(frame,(button_x1,button_y1),(button_x2,button_y2),(0,0,255),-1)

    cv2.putText(frame,"EXIT",
                (button_x1+40,button_y1+40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2)

    # check exit
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            for lm in hand_landmarks.landmark:

                px = int(lm.x*w)
                py = int(lm.y*h)

                if button_x1 < px < button_x2 and button_y1 < py < button_y2:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    cv2.imshow("Pour Water",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()