import cv2
import mediapipe as mp

def main():
    # Initialize OpenCV webcam
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,            # Track only 1 hand for simplicity
        min_detection_confidence=0.7, # Minimum confidence to detect a hand
        min_tracking_confidence=0.7   # Minimum confidence to track a hand
    )
    mp_draw = mp.solutions.drawing_utils

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Hand tracking active! Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame.")
            break

        # Flip frame horizontally for natural mirror movement
        frame = cv2.flip(frame, 1)
        
        # OpenCV reads frames in BGR format, but MediaPipe expects RGB. 
        # We must convert the color space before processing.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        result = hands.process(rgb_frame)

        # Check if any hands are detected in the frame
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the hand skeleton connections on our video frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Example: Let's extract the coordinates of landmark #8 (Index Finger Tip)
                h, w, c = frame.shape
                index_finger_tip = hand_landmarks.landmark[8]
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Draw a small blue circle on the index finger tip
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        # Display the output
        cv2.imshow('Air Canvas - Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()