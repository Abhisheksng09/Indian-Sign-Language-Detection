import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict=pickle.load(open('model.p','rb'))
model=model_dict['model']

cap=cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,3200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1800)


mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles

hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.2)

labels_dict={0:'1', 1:'A', 2:'B'}

while True:
    data_auxi1=[]
    x_=[]
    y_=[]

    desired_length=84

    ret, frame=cap.read()
    frame=cv2.flip(frame,1)

    H, W, _=frame.shape

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results=hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, # image to draw
                hand_landmarks, # model output
                mp_hands.HAND_CONNECTIONS, # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    data_auxi1.append(x)
                    data_auxi1.append(y)
                    x_.append(x)
                    y_.append(y)

        x1=int(min(x_) * W)-10
        y1=int(min(y_) * H)-10

        x2=int(max(x_) * W)-10
        y2=int(max(y_) * H)-10
        
        while len(data_auxi1)<84:
            data_auxi1.extend([0.0,0.0])

        prediction=model.predict([np.asarray(data_auxi1)])

        predicted_character=labels_dict[int(float(prediction[0]))]

        # print(predicted_character)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),4)
        cv2.putText(frame,predicted_character,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),3,cv2.LINE_AA)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('d'):
        break

cap.release()
cap.destroyAllWindows()