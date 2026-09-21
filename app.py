import cv2
import mediapipe as mp
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    # Variables to track previous finger coordinates for continuous lines
    xp, yp = 0, 0
    paint_color = (255, 0, 255)  # Default drawing color: Magenta (BGR format)
    brush_thickness = 5

    # Initialize a blank canvas variable as None; we will size it using the first video frame
    canvas = None

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Air Canvas active! Click inside the video window and press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Initialize the black drawing canvas matching the webcam frame size
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract Index Finger Tip (Landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Draw cursor circle on live feed
                cv2.circle(frame, (cx, cy), 10, paint_color, cv2.FILLED)

                # Basic drawing logic: If previous points are zero, set them to current point
                if xp == 0 and yp == 0:
                    xp, yp = cx, cy

                # Draw a line from previous point to current point on our persistent canvas
                cv2.line(canvas, (xp, yp), (cx, cy), paint_color, brush_thickness)
                
                # Update previous coordinates
                xp, yp = cx, cy
        else:
            # Reset previous points if hand leaves the frame so lines don't streak across the screen
            xp, yp = 0, 0

        # Merge the drawing canvas with the live webcam frame using bitwise operations
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
        
        frame = cv2.bitwise_and(frame, inv_canvas)
        frame = cv2.bitwise_or(frame, canvas)

        # Display output window
        cv2.imshow('Air Canvas', frame)

        # Make sure to click inside the 'Air Canvas' video window before pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()