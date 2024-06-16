import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pymycobot.mycobot import MyCobot
import time
import os

mc = MyCobot('COM3', 115200)

i = 0
x = 0
w = 0

this_action = '?'
cobot_run = False
actions = ['1','2','3','4']

seq_length = 30

model = load_model('opencv4_hand_AI_robotarm_motion\model.h5')



# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

seq = []
action_seq = []

def init():
    mc.send_angles([0,0,0,0,0,0],40)
    time.sleep(4)
    
def zero():
    mc.send_angles([-90,0,-40,-50,90,4],40)
    time.sleep(2)
    
def gripper_open():
    print("OPEN")
    mc.set_eletric_gripper(0)   #그리퍼 현 상태를 닫았다고 인식
    mc.set_gripper_value(100,30)    
    time.sleep(2)

def gripper_close():
    print("CLOSE")
    mc.set_eletric_gripper(1)
    mc.set_gripper_value(16,30) #16까지 닫기
    time.sleep(2)
    
def go_belt_up():
    mc.send_angles([-90,0,-40,-30,93,4],40)
    time.sleep(2)
    
def go_belt_down():
    mc.send_angles([-90,-19,-49,-28,93,4],40)
    time.sleep(2)
    
def middle_point():
    mc.send_angles([0,0,-40,-30,90,4],40)
    time.sleep(2)
    
#block_orange
def bring_orange_three():
    mc.send_angles([80,0,-40,-30,90,4],40)  #orange 색상으로 이동
    time.sleep(4)
    mc.send_angles([80,-39,-47,3,90,10],40) #고개 down
    time.sleep(4)
    gripper_close()
    mc.send_angles([80,0,-40,-30,90,4],40)  #3층의 블록을 잡아서 up
    time.sleep(4)
    middle_point()
    go_belt_up()    #인식장소로 이동
    gripper_open()
    
def bring_orange_two():
    mc.send_angles([80,0,-40,-30,90,4],40)  #orange 색상으로 이동
    time.sleep(4)
    mc.send_angles([80,-43,-54,12,90,5],40) #고개 down
    time.sleep(4)
    gripper_close()
    mc.send_angles([80,0,-40,-30,90,4],40)  #2층의 블록을 잡아서 up
    time.sleep(4)
    middle_point()
    go_belt_up()    #인식장소로 이동
    gripper_open()
    
def bring_orange_one():
    mc.send_angles([80,0,-40,-30,90,4],40)  #orange 색상으로 이동
    time.sleep(4)
    mc.send_angles([80,-48,-53,15,90,0],40) #고개 down
    time.sleep(4)
    gripper_close()
    mc.send_angles([80,0,-40,-30,90,4],40)  #1층의 블록을 잡아서 up
    time.sleep(4)
    middle_point()
    go_belt_up()    #인식장소로 이동
    gripper_open()


#block_green
def bring_green_three():
    mc.send_angles([100,0,-40,-30,90,4],40) #green 색상으로 이동
    time.sleep(2)
    mc.send_angles([100,-38,-54,10,90,8],40)    #고개 down
    time.sleep(2)
    gripper_close()
    mc.send_angles([100,0,-40,-30,90,4],40) #3층의 블록을 잡아서 up
    time.sleep(2)
    middle_point()
    go_belt_up()
    gripper_open()
    
def bring_green_two():
    mc.send_angles([100,0,-40,-30,90,4],40) #green 색상으로 이동
    time.sleep(2)
    mc.send_angles([100,-43,-56,16,90,5],40)    #고개 down
    time.sleep(2)
    gripper_close()
    mc.send_angles([100,0,-40,-30,90,4],40) #2층의 블록을 잡아서 up
    time.sleep(2)
    middle_point()
    go_belt_up()
    gripper_open()
    
def bring_green_one():
    mc.send_angles([100,0,-40,-30,90,4],40) #green 색상으로 이동
    time.sleep(2)
    mc.send_angles([100,-50,-52,16,90,0],40)    #고개 down
    time.sleep(2)
    gripper_close()
    mc.send_angles([100,0,-40,-30,90,4],40) #1층의 블록을 잡아서 up
    time.sleep(2)
    middle_point()
    go_belt_up()
    gripper_open()
    
    
#block_yellow
def bring_yellow_three():
    mc.send_angles([120,0,-40,-30,90,4],40) #yellow 색상으로 이동
    time.sleep(2)
    mc.send_angles([120,-38,-52,6,90,2],40)    #고개 down
    time.sleep(2)
    gripper_close()
    mc.send_angles([120,0,-40,-30,90,4],40) #3층의 블록을 잡아서 up
    time.sleep(2)
    middle_point()
    go_belt_up()
    gripper_open()
    
def bring_yellow_two():
    mc.send_angles([120,0,-40,-30,90,4],40) #yellow 색상으로 이동
    time.sleep(2)
    mc.send_angles([120,-43,-52,10,90,0],40)    #고개 down
    time.sleep(2)
    gripper_close()
    mc.send_angles([120,0,-40,-30,90,4],40) #2층의 블록을 잡아서 up
    time.sleep(2)
    middle_point()
    go_belt_up()
    gripper_open()
    
def bring_yellow_one():
    mc.send_angles([120,0,-40,-30,90,4],40) #yellow 색상으로 이동
    time.sleep(2)
    mc.send_angles([120,-48,-53,15,90,0],40)    #고개 down
    time.sleep(2)
    gripper_close()
    mc.send_angles([120,0,-40,-30,90,4],40) #1층의 블록을 잡아서 up
    time.sleep(2)
    middle_point()
    go_belt_up()
    gripper_open()
    
def operate_orange():
    global i
    gripper_open()
    
    if i==0:
        
        init()
        bring_orange_three()
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero()
        print("first_orange_operation_clear")

    elif i==1:
        init()
        bring_orange_two()
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero()
        print("second_orange_operation_clear")

    elif i==2:
        
        init()
        bring_orange_one()
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero()
        print("third_orange_operation_clear")
        print("All operations clear")

    else :
        print("warning_it_isn't_pyellowictable")
        go_belt_up()
        
def operate_yellow():
    global x
    gripper_open()
    
    if x==0:
        
        init()
        bring_yellow_three()
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero()
        print("first_yellow_operation_clear")

    elif x==1:
        
        init()
        bring_yellow_two()
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero()
        print("second_yellow_operation_clear")

    elif x==2:
    
        init()
        bring_yellow_one()
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero()
        print("third_yellow_operation_clear")
        print("All operations clear")

    else :
        print("warning_it_isn't_pyellowictable")
        go_belt_up()       

def operate_green():
    global w
    gripper_open()
    
    if w==0:
        
        init()
        bring_green_three()
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero()
        print("first_green_operation_clear")

    elif w==1:
        
        init()
        bring_green_two()
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero()
        print("second_green_operation_clear")

    elif w==2:
       
        init()
        bring_green_one()
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero()
        print("third_green_operation_clear")
        print("All operations clear")

    else :
        print("warning_it_isn't_pyellowictable")
        go_belt_up()
        
        

gripper_open()
zero()
mc.set_gripper_mode(0)
mc.init_eletric_gripper()
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
            operate_orange()
            i=i+1
            i=i%3
        
        if this_action == '2':
            operate_yellow()
            x=x+1
            x=x%3

        if this_action == '3':
            operate_green()
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

