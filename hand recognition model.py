import cv2
import mediapipe as mp

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils

def draw_grid(image, rows=3, cols=3):
    h,w,_=image.shape
    
    for i in range(1, rows):
        y=int(i*h/rows)
        cv2.line(image, (0,y), (w,y), (255,255,255), 2)
    
    for j in range(1,cols):
        x=int(j*w/cols)
        cv2.line(image, (x,0), (x,h), (255,255,255), 2)
    
cap=cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success,image=cap.read()
        if not success:
            break

        image=cv2.flip(image,1)
        
        image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        results=hands.process(image_rgb)
        
        draw_grid(image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks,mp_hands.HAND_CONNECTIONS)
            
        cv2.imshow('Hand Grid', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
