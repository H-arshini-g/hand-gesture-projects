import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Print landmark positions
            for id, landmark in enumerate(hand_landmarks.landmark):

                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)

                # Show landmark ID
                cv2.putText(
                    frame,
                    str(id),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,255,0),
                    1
                )

    # Show camera feed
    cv2.imshow("Hand Landmark Visualizer", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()