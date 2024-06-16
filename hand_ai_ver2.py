import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pymycobot.mycobot import MyCobot
import time
import os
import cobot_function as cf

mc = MyCobot('COM3', 115200)

i = 0
x = 0
w = 0

this_action = '?'
cobot_run = False
actions = ['1','2','3','4']

seq_length = 30

model = load_model('model.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

seq = []
action_seq = []

if __name__ == "__main__":
    cf.gripper_open(mc)
    cf.zero(mc)
    cf.mc.set_gripper_mode(0)
    cf.mc.init_eletric_gripper()
    time.sleep(1)
    
        
    webcam_video = cv2.VideoCapture(0)

    while webcam_video.isOpened():
        
        success, video = webcam_video.read()
        
        img = cv2.flip(video, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        hands_err = ''

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.95:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 8:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]== action_seq[-4]== action_seq[-5]== action_seq[-6]== action_seq[-7]== action_seq[-8]:
                    this_action = action

                cv2.putText(img, this_action , (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        
        else:
            hands_err = 'show hand'
            cv2.putText(img, hands_err , (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            this_action = '?'
            action_seq.clear()
            seq.clear()

        if this_action != '?':
            cobot_run = True
            if this_action == '1':
                cf.operate_orange()
                i=i+1
                i=i%3
            
            if this_action == '2':
                cf.operate_yellow()
                x=x+1
                x=x%3

            if this_action == '3':
                cf.operate_green()
                w=w+1
                w=w%3
            this_action = '?'
            action_seq.clear()
            seq.clear()

        cv2.imshow("window",img)
        cobot_run = False
        if cv2.waitKey(1) & 0xFF == 27 :
            break
        
    webcam_video.release()

