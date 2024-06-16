import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pymycobot.mycobot import MyCobot
import time

class MyCobotController:
    def __init__(self, port='COM3', baudrate=115200):
        self.mc = MyCobot(port, baudrate)
        self.init()

    def initi(self):
        self.mc.send_angles([0,0,0,0,0,0],40)
        time.sleep(4)
    
    def zero(self):
        self.mc.send_angles([-90,0,-40,-50,90,4],40)
        time.sleep(2)
    
    def gripper_open(self):
        print("OPEN")
        self.mc.set_eletric_gripper(0)
        self.mc.set_gripper_value(100,30)
        time.sleep(2)

    def gripper_close(self):
        print("CLOSE")
        self.mc.set_eletric_gripper(1)
        self.mc.set_gripper_value(16,30)
        time.sleep(2)
        
    def go_belt_up(self):
        self.mc.send_angles([-90,0,-40,-30,93,4],40)
        time.sleep(2)
        
    def go_belt_down(self):
        self.mc.send_angles([-90,-19,-49,-28,93,4],40)
        time.sleep(2)
        
    def middle_point(self):
        self.mc.send_angles([0,0,-40,-30,90,4],40)
        time.sleep(2)
        
    #block_orange
    def bring_orange_three(self):
        self.mc.send_angles([80,0,-40,-30,90,4],40)  #orange 색상으로 이동
        time.sleep(4)
        self.mc.send_angles([80,-39,-47,3,90,10],40) #고개 down
        time.sleep(4)
        self.gripper_close()
        self.mc.send_angles([80,0,-40,-30,90,4],40)  #3층의 블록을 잡아서 up
        time.sleep(4)
        self.middle_point()
        self.go_belt_up()    #인식장소로 이동
        self.gripper_open()
        
    def bring_orange_two(self):
        self.mc.send_angles([80,0,-40,-30,90,4],40)  #orange 색상으로 이동
        time.sleep(4)
        self.mc.send_angles([80,-43,-54,12,90,5],40) #고개 down
        time.sleep(4)
        self.gripper_close()
        self.mc.send_angles([80,0,-40,-30,90,4],40)  #2층의 블록을 잡아서 up
        time.sleep(4)
        self.middle_point()
        self.go_belt_up()    #인식장소로 이동
        self.gripper_open()
        
    def bring_orange_one(self):
        self.mc.send_angles([80,0,-40,-30,90,4],40)  #orange 색상으로 이동
        time.sleep(4)
        self.mc.send_angles([80,-48,-53,15,90,0],40) #고개 down
        time.sleep(4)
        self.gripper_close()
        self.mc.send_angles([80,0,-40,-30,90,4],40)  #1층의 블록을 잡아서 up
        time.sleep(4)
        self.middle_point()
        self.go_belt_up()    #인식장소로 이동
        self.gripper_open()


    #block_green
    def bring_green_three(self):
        self.mc.send_angles([100,0,-40,-30,90,4],40) #green 색상으로 이동
        time.sleep(2)
        self.mc.send_angles([100,-38,-54,10,90,8],40)    #고개 down
        time.sleep(2)
        self.gripper_close()
        self.mc.send_angles([100,0,-40,-30,90,4],40) #3층의 블록을 잡아서 up
        time.sleep(2)
        self.middle_point()
        self.go_belt_up()
        self.gripper_open()
        
    def bring_green_two(self):
        self.mc.send_angles([100,0,-40,-30,90,4],40) #green 색상으로 이동
        time.sleep(2)
        self.mc.send_angles([100,-43,-56,16,90,5],40)    #고개 down
        time.sleep(2)
        self.gripper_close()
        self.mc.send_angles([100,0,-40,-30,90,4],40) #2층의 블록을 잡아서 up
        time.sleep(2)
        self.middle_point()
        self.go_belt_up()
        self.gripper_open()
        
    def bring_green_one(self):
        self.mc.send_angles([100,0,-40,-30,90,4],40) #green 색상으로 이동
        time.sleep(2)
        self.mc.send_angles([100,-50,-52,16,90,0],40)    #고개 down
        time.sleep(2)
        self.gripper_close()
        self.mc.send_angles([100,0,-40,-30,90,4],40) #1층의 블록을 잡아서 up
        time.sleep(2)
        self.middle_point()
        self.go_belt_up()
        self.gripper_open()
        
        
    #block_yellow
    def bring_yellow_three(self):
        self.mc.send_angles([120,0,-40,-30,90,4],40) #yellow 색상으로 이동
        time.sleep(2)
        self.mc.send_angles([120,-38,-52,6,90,2],40)    #고개 down
        time.sleep(2)
        self.gripper_close()
        self.mc.send_angles([120,0,-40,-30,90,4],40) #3층의 블록을 잡아서 up
        time.sleep(2)
        self.middle_point()
        self.go_belt_up()
        self.gripper_open()
        
    def bring_yellow_two(self):
        self.mc.send_angles([120,0,-40,-30,90,4],40) #yellow 색상으로 이동
        time.sleep(2)
        self.mc.send_angles([120,-43,-52,10,90,0],40)    #고개 down
        time.sleep(2)
        self.gripper_close()
        self.mc.send_angles([120,0,-40,-30,90,4],40) #2층의 블록을 잡아서 up
        time.sleep(2)
        self.middle_point()
        self.go_belt_up()
        self.gripper_open()
        
    def bring_yellow_one(self):
        self.mc.send_angles([120,0,-40,-30,90,4],40) #yellow 색상으로 이동
        time.sleep(2)
        self.mc.send_angles([120,-48,-53,15,90,0],40)    #고개 down
        time.sleep(2)
        self.gripper_close()
        self.mc.send_angles([120,0,-40,-30,90,4],40) #1층의 블록을 잡아서 up
        time.sleep(2)
        self.middle_point()
        self.go_belt_up()
        self.gripper_open()
        
    def operate_orange(self):
        global i
        self.gripper_open()
        
        if i==0:
            
            self.initi()
            self.bring_orange_three()
            # go_stack_first_point_orange_two()
            # go_stack_first_point_orange_three()
            self.zero()
            print("first_orange_operation_clear")

        elif i==1:
            self.initi()
            self.bring_orange_two()
            # go_stack_first_point_orange_two()
            # go_stack_first_point_orange_three()
            self.zero()
            print("second_orange_operation_clear")

        elif i==2:
            
            self.initi()
            self.bring_orange_one()
            # go_stack_first_point_orange_two()
            # go_stack_first_point_orange_three()
            self.zero()
            print("third_orange_operation_clear")
            print("All operations clear")

        else :
            print("warning_it_isn't_pyellowictable")
            self.go_belt_up()
            
    def operate_yellow(self):
        global x
        self.gripper_open()
        
        if x==0:
            
            self.initi()
            self.bring_yellow_three()
            # go_stack_first_point_orange_two()
            # go_stack_first_point_orange_three()
            self.zero()
            print("first_yellow_operation_clear")

        elif x==1:
            
            self.initi()
            self.bring_yellow_two()
            # go_stack_first_point_orange_two()
            # go_stack_first_point_orange_three()
            self.zero()
            print("second_yellow_operation_clear")

        elif x==2:
        
            self.initi()
            self.bring_yellow_one()
            # go_stack_first_point_orange_two()
            # go_stack_first_point_orange_three()
            self.zero()
            print("third_yellow_operation_clear")
            print("All operations clear")

        else :
            print("warning_it_isn't_pyellowictable")
            self.go_belt_up()       

    def operate_green(self):
        global w
        self.gripper_open()
        
        if w==0:
            
            self.initi()
            self.bring_green_three()
            # go_stack_first_point_orange_two()
            # go_stack_first_point_orange_three()
            self.zero()
            print("first_green_operation_clear")

        elif w==1:
            
            self.initi()
            self.bring_green_two()
            # go_stack_first_point_orange_two()
            # go_stack_first_point_orange_three()
            self.zero()
            print("second_green_operation_clear")

        elif w==2:
        
            self.initi()
            self.bring_green_one()
            # go_stack_first_point_orange_two()
            # go_stack_first_point_orange_three()
            self.zero()
            print("third_green_operation_clear")
            print("All operations clear")

        else :
            print("warning_it_isn't_pyellowictable")
            self.go_belt_up()
            
class HandTracking:
    def __init__(self, model_path='model.h5', seq_length=30, video_source=0):
        self.model = load_model(model_path)
        self.seq_length = seq_length
        self.video_source = video_source
        self.cobot_controller = MyCobotController()


        # MediaPipe hands model
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.seq = []
        self.actions = ['1', '2', '3', '4']
        self.action_seq = []

    def run_hand_tracking(self):
        webcam_video = cv2.VideoCapture(self.video_source)

        while webcam_video.isOpened():
            success, video = webcam_video.read()

            img = cv2.flip(video, 1)
            img_before = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.hands.process(img_before)
            img_after = cv2.cvtColor(img_before, cv2.COLOR_RGB2BGR)

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

                    self.seq.append(d)

                    self.mp_drawing.draw_landmarks(img_after, res, self.mp_hands.HAND_CONNECTIONS)

                    if len(self.seq) < self.seq_length:
                        continue

                    input_data = np.expand_dims(np.array(self.seq[-self.seq_length:], dtype=np.float32), axis=0)

                    y_pred = self.model.predict(input_data).squeeze()

                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]

                    if conf < 0.95:
                        continue

                    action = self.actions[i_pred]
                    self.action_seq.append(action)

                    if len(self.action_seq) < 8:
                        continue

                    this_action = '?'
                    if (self.action_seq[-1] == self.action_seq[-2] == self.action_seq[-3] ==
                        self.action_seq[-4] == self.action_seq[-5] == self.action_seq[-6] ==
                        self.action_seq[-7] == self.action_seq[-8]):
                        this_action = action

                    cv2.putText(img_after, this_action , (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            else:
                hands_err = 'show hand'
                cv2.putText(img_after, hands_err , (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                this_action = '?'
                self.action_seq.clear()
                self.seq.clear()

            if this_action != '?':
                cobot_run = True
                if this_action == '1':
                    self.cobot_controller.operate_orange()
                    i=i+1
                    i=i%3
                
                if this_action == '2':
                    self.cobot_controller.operate_yellow()
                    x=x+1
                    x=x%3

                if this_action == '3':
                    self.cobot_controller.operate_green()
                    w=w+1
                    w=w%3
                this_action = '?'
                self.action_seq.clear()
                self.seq.clear()
                
            cv2.imshow("window",img)
            cobot_run = False
            if cv2.waitKey(1) & 0xFF == 27 :
                break

        webcam_video.release()

if __name__ == "__main__":
    cobot_controller = MyCobotController()
    hand_tracking = HandTracking()
    
    
