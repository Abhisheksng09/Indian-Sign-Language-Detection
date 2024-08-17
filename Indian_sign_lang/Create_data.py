import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles

hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.2)

DATA_DIR='F:\\Sign detection\\dataset\\data'

data=[]
labels=[]

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        data_auxi1=[]
        img=cv2.imread(os.path.join(DATA_DIR,dir_,img_path))
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        results=hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    data_auxi1.append(x)
                    data_auxi1.append(y)

            data.append(data_auxi1)
            labels.append(dir_)

# print(data)
# print(labels)

f=open('data.pickle1','wb')
pickle.dump({'data':data,'labels':labels},f)
f.close()