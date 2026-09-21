import cv2
import mediapipe as mp
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )
    mp_draw = mp.solutions.drawing_utils

    xp, yp = 0, 0
    paint_color = (255, 0, 255)  # Default: Magenta
    brush_thickness = 8
    eraser_thickness = 40

    canvas = None

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Air Canvas with Gestures active! Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw UI Header Bar on the live frame
        cv2.rectangle(frame, (0, 0), (w, 100), (50, 50, 50), -1)
        cv2.putText(frame, "CLEAR", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(frame, "BLUE", (250, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(frame, "GREEN", (450, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, "ERASER", (650, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get Index finger tip (8) and Middle finger tip (12)
                x1, y1 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                x2, y2 = int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h)

                # Check which fingers are up (simplified: comparing y coordinates of tips vs lower joints)
                # If index is up and middle is down -> Drawing Mode
                # If both are up -> Selection Mode
                index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
                middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y

                if index_up and middle_up:
                    xp, yp = 0, 0 # Reset previous points
                    cv2.circle(frame, (x1, y1), 15, (0, 255, 255), cv2.FILLED) # Selection cursor
                    
                    # Selection Logic if hand is in the header area (y < 100)
                    if y1 < 100:
                        if 30 < x1 < 180:
                            canvas = np.zeros((h, w, 3), dtype=np.uint8) # Clear Canvas
                        elif 230 < x1 < 380:
                            paint_color = (255, 0, 0) # Blue
                        elif 430 < x1 < 580:
                            paint_color = (0, 255, 0) # Green
                        elif 630 < x1 < 780:
                            paint_color = (0, 0, 0) # Eraser (Black)

                elif index_up and not middle_up:
                    # Drawing Mode
                    cv2.circle(frame, (x1, y1), 10, paint_color, cv2.FILLED)
                    
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    current_thickness = eraser_thickness if paint_color == (0, 0, 0) else brush_thickness
                    cv2.line(canvas, (xp, yp), (x1, y1), paint_color, current_thickness)
                    xp, yp = x1, y1
                else:
                    xp, yp = 0, 0
        else:
            xp, yp = 0, 0

        # Merge canvas with frame
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
        
        frame = cv2.bitwise_and(frame, inv_canvas)
        frame = cv2.bitwise_or(frame, canvas)

        cv2.imshow('Air Canvas', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()