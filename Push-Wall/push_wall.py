import cv2
import mediapipe as mp
import time

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

cv2.namedWindow("Push The Wall", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Push The Wall", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

push_start = None
push_duration = 5
score = 0

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    palm_center = None

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark[9]

            x = int(lm.x * w)
            y = int(lm.y * h)

            palm_center = (x,y)

            cv2.circle(frame,(x,y),10,(255,0,255),-1)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # wall area
    wall_x1 = w//2 - 100
    wall_y1 = h//2 - 100
    wall_x2 = w//2 + 100
    wall_y2 = h//2 + 100

    cv2.rectangle(frame,(wall_x1,wall_y1),(wall_x2,wall_y2),(0,255,255),3)

    pushing = False

    if palm_center:

        if wall_x1 < palm_center[0] < wall_x2 and wall_y1 < palm_center[1] < wall_y2:
            pushing = True

    if pushing:

        if push_start is None:
            push_start = time.time()

        elapsed = time.time() - push_start

        cv2.putText(frame,
                    f"Pushing: {int(elapsed)} / {push_duration}",
                    (50,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        if elapsed >= push_duration:
            score += 1
            push_start = None

    else:
        push_start = None

    cv2.putText(frame,
                f"Score: {score}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    # EXIT button
    button_x1 = w - 180
    button_y1 = 20
    button_x2 = w - 20
    button_y2 = 80

    cv2.rectangle(frame,(button_x1,button_y1),(button_x2,button_y2),(0,0,255),-1)

    cv2.putText(frame,
                "EXIT",
                (button_x1+40,button_y1+40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2)

    if palm_center:
        if button_x1 < palm_center[0] < button_x2 and button_y1 < palm_center[1] < button_y2:
            break

    cv2.imshow("Push The Wall",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()