import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# object position
obj_x = 300
obj_y = 300
radius = 40

dragging = False

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            cv2.circle(frame, (x1,y1), 10, (255,0,0), -1)
            cv2.circle(frame, (x2,y2), 10, (0,255,0), -1)

            distance = math.hypot(x2-x1, y2-y1)

            # detect pinch
            if distance < 40:

                if math.hypot(obj_x-x2, obj_y-y2) < radius:
                    dragging = True

            else:
                dragging = False

            # move object
            if dragging:
                obj_x = x2
                obj_y = y2

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # draw object
    cv2.circle(frame,(obj_x,obj_y),radius,(0,0,255),-1)

    cv2.imshow("Drag Object Demo",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()