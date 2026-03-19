import cv2
import mediapipe as mp
import math

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            h, w, _ = frame.shape

            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            # Draw fingertip points
            cv2.circle(frame, (x1,y1), 10, (255,0,0), -1)
            cv2.circle(frame, (x2,y2), 10, (0,255,0), -1)

            # Draw line between fingers
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 2)

            # Distance between thumb and index
            distance = math.hypot(x2-x1, y2-y1)

            # Display distance
            cv2.putText(frame,f"Distance: {int(distance)}",
                        (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),2)

            # Detect pinch
            if distance < 40:
                cv2.putText(frame,"PINCH DETECTED",
                            (20,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,0,255),3)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Pinch Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()