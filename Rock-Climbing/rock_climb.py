import cv2
import mediapipe as mp
import random
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# set higher camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

score = 0
hold_x = None
hold_y = None
step = 40

holds = []

# create fullscreen window
cv2.namedWindow("Rock Climbing", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Rock Climbing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    h, w, _ = frame.shape

    if hold_x is None:
        hold_x = w // 2
        hold_y = h - 120

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingertip = None

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark[8]

            x = int(lm.x * w)
            y = int(lm.y * h)

            fingertip = (x,y)

            cv2.circle(frame,(x,y),10,(255,0,255),-1)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # draw previous holds
    for hx,hy in holds:
        cv2.circle(frame,(hx,hy),25,(255,0,0),-1)

    # draw current hold
    cv2.circle(frame,(hold_x,hold_y),30,(0,255,0),-1)

    if fingertip:

        dist = math.hypot(
            fingertip[0]-hold_x,
            fingertip[1]-hold_y
        )

        if dist < 40:

            score += 1
            holds.append((hold_x,hold_y))

            hold_y -= step
            hold_x = random.randint(100, w-100)

            if hold_y < 120:
                hold_y = h - 120
                holds.clear()

    # dynamic exit button (top-right)
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

    # detect fingertip touching exit button
    if fingertip:
        if button_x1 < fingertip[0] < button_x2 and button_y1 < fingertip[1] < button_y2:
            break

    # score
    cv2.putText(frame,
                f"Score: {score}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    cv2.imshow("Rock Climbing",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()