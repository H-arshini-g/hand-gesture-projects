import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random

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

cv2.namedWindow("Puzzle Game", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Puzzle Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Load puzzle image
image = cv2.imread("images/puzzle.jpeg")
image = cv2.resize(image,(300,300))

rows = 3
cols = 3

piece_h = image.shape[0]//rows
piece_w = image.shape[1]//cols

pieces = []

for r in range(rows):
    for c in range(cols):

        piece = image[r*piece_h:(r+1)*piece_h, c*piece_w:(c+1)*piece_w]

        x = random.randint(50,500)
        y = random.randint(50,400)

        pieces.append({
            "img":piece,
            "x":x,
            "y":y,
            "correct_x":c*piece_w+700,
            "correct_y":r*piece_h+100,
            "placed":False
        })

dragging_piece = None

start_time = time.time()
time_limit = 30

smooth_x = 0
smooth_y = 0

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    pinch = False
    fx,fy = 0,0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            x1,y1 = int(thumb.x*w), int(thumb.y*h)
            x2,y2 = int(index.x*w), int(index.y*h)

            smooth_x = int(0.7*smooth_x + 0.3*x2)
            smooth_y = int(0.7*smooth_y + 0.3*y2)

            fx, fy = smooth_x, smooth_y

            dist = math.hypot(x2-x1,y2-y1)

            if dist < 40:
                pinch = True

            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

    # drag logic
    if pinch and dragging_piece is None:

        for piece in pieces:

            if piece["placed"]:
                continue

            px, py = piece["x"], piece["y"]

            if abs(fx - px) < piece_w and abs(fy - py) < piece_h:
                dragging_piece = piece
                break

    if not pinch:
        dragging_piece = None

    if dragging_piece:
        dragging_piece["x"] = fx-piece_w//2
        dragging_piece["y"] = fy-piece_h//2

    # snapping
    for piece in pieces:

        if not piece["placed"]:

            distance = math.hypot(
                piece["x"] - piece["correct_x"],
                piece["y"] - piece["correct_y"]
            )

            if distance < 70:
                piece["x"] = piece["correct_x"]
                piece["y"] = piece["correct_y"]
                piece["placed"] = True

    # draw pieces
    for piece in pieces:

        x, y = piece["x"], piece["y"]

        if 0 <= x < w-piece_w and 0 <= y < h-piece_h:

            frame[y:y+piece_h, x:x+piece_w] = piece["img"]

            if piece is dragging_piece:
                cv2.rectangle(frame,(x,y),(x+piece_w,y+piece_h),(0,255,0),3)

            if piece["placed"]:
                cv2.rectangle(frame,(x,y),(x+piece_w,y+piece_h),(255,0,0),3)

    # timer
    remaining = max(0, int(time_limit-(time.time()-start_time)))

    cv2.putText(frame,f"Time: {remaining}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    # win check
    if all(p["placed"] for p in pieces):

        cv2.putText(frame,"PUZZLE COMPLETE",
                    (400,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0,255,0),
                    3)

    # EXIT button
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

    # gesture exit
    if button_x1 < fx < button_x2 and button_y1 < fy < button_y2:
        break

    cv2.imshow("Puzzle Game",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()