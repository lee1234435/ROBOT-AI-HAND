import cv2, threading, time
import numpy as np
import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
# model = YOLO('yolov8n.pt')
from pymycobot.mycobot import MyCobot

mc = MyCobot('COM5',115200)
is_quit = False
direction, color, color_hand = None, None, None
red_cnt, green_cnt, blue_cnt = 0, 0, 0

# model = YOLO('yolo_final/best.pt')
model = YOLO('./MyCobot/best.pt')
print(type(model.names),len(model.names), model.names)
names = model.model.names

from collections import defaultdict
track_history = defaultdict(lambda: [])

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)

def color_detect():
    global color
    if color == 0:
        color = 'blue'
        # print("Detected color: blue")
    elif color == 1:
        color = 'green'
        # print("Detected color: green")
    elif color == 2:
        color = 'orange'
        # print("Detected color: orange")
    elif color == 3:
        color = 'red'
        # print("Detected color: red")
    elif color == 4:
        color = 'yellow'
        # print("Detected color: yellow")
    else:
        color = "No"


def camera_thread(cap):
    global direction, color

    while cap.isOpened():
        success, frame = cap.read()
        height, width, _ = frame.shape
        
        if success:
            results = model.track(frame, persist=True, verbose=False) # model.predict 대신
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:

                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()

                # Annotator Init
                annotator = Annotator(frame, line_width=2)

                for box, cls, track_id in zip(boxes, clss, track_ids):
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
                    color = int(cls)
                    
                    # 중심 좌표 계산
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)

                    # 중심점 표시
                    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                    color_detect()

                center_camera_x = width // 2
                center_camera_y = height // 2

                # direction = front | backward | good
                if center_y < center_camera_y + 180 - 10:
                    direction = "front"
                    # print(cy, center_line1, height)
                    # print('front')
                elif center_y > center_camera_y + 200 - 10:
                    direction = "backward"
                    # print(cy, center_line1, height)
                    # print('backward')#240 459
                else:
                    direction = "good"
                    # print("good!")
                
                # color_detect()
            

            if cv2.waitKey(1) & 0xFF == ord("x"):
                is_quit = True
                time.sleep(0.5)
                break
            
            cv2.imshow("yolo", frame)
        else:
            break



step = 10
steps = {
    0: '[컨베이어] 컨베이어 벨트로 이동',
    1: '[컨베이어] 블럭 색상 판별 - color 값에 따라, red_cnt, green_cnt, blue_cnt 값 세팅',
    2: '[컨베이어] 블럭 이동 대기(컨베이어 벨트에서 이동하는 시간 동안 대기)',
    3: '[컨베이어] x축 조정 - direction 값에 따라',
    4: '[컨베이어] 잡기',
    5: '[중간지점] 중간지점으로 이동',
    6: '[블록무더기] 무더기별 시작위치로 이동 - color 값에 따라 이동 위치 결정',
    7: '[블록무더기] z축 조정 - xxx_cnt 값에 따라 z축 조정',
    8: '[블록무더기] 놓기',
    9: '[블록무더기] 무더기별 시작위치로 이동 - color 값에 따라 이동 위치 결정',    
    10: '[중간지점] 중간지점으로 이동',
}


def z_height(cnt):
    z_move = 0
    match cnt:
        case 1:
            z_move = 70
        case 2:
            # z_move = 40
            z_move = 45
        case 3:
            # z_move = 10
            z_move = 20
        case _:
            z_move = 0
    print(z_move)
    return z_move
    

def big_move(target):
    global mode, angles_target, x,y,z,rx,ry,rz
    mode = 'angles'
    angles_target = target
    time.sleep(1.5)
    print("이동 완료")
    x,y,z,rx,ry,rz = mc.get_coords()
    mode = 'standby'
    
    
def nextstep_thread():
    global step, mode, x,y,z,rx,ry,rz, angles_target, is_quit, color, color_hand, direction, red_cnt, green_cnt, blue_cnt
    
    y_axis_cnt = 0
    
    while True:
        if is_quit:
            break
                
        match step:
            
            case 0: 
                print("### 스텝 0 : [컨베이어] 컨베이어 벨트로 이동")
                big_move('conveyor')
                color = 'No'
                step += 1
                
            case 1:
                print("### 스텝 1 : [컨베이어] 블럭 색상 판별 - color 값에 따라, red_cnt, green_cnt, blue_cnt 값 세팅")
                if color in ['red', 'orange', 'green', 'blue']:
                    if color == 'red' or color == 'orange':
                        red_cnt += 1
                    if color == 'green':
                        green_cnt += 1
                    if color == 'blue':
                        blue_cnt += 1
                    color_hand = color
                    
                    if color_hand == 'orange':
                        color_hand = 'red'
                    if red_cnt >= 4:
                        red_cnt = 1
                    if green_cnt >= 4:
                        green_cnt = 1
                    if blue_cnt >= 4:
                        blue_cnt = 1
                        
                    step += 1
                else:
                    time.sleep(0.5)
                
            case 2:
                print("### 스텝 2 : [컨베이어] 블럭 이동 대기(컨베이어 벨트에서 이동하는 시간 동안 대기)")
                print(f"COLOR:{color_hand}, R:{red_cnt}, G:{green_cnt}, B:{blue_cnt}")
                print("대기...")
                time.sleep(2.5)
                step += 1
                
            case 3:
                print("### 스텝 3 : [컨베이어] y축 조정 - direction 값에 따라")
                print("y축 조정: ", direction, "y: ", y)
                mode = 'coords'
                if direction == 'front':
                    y += 2.5
                    if y > 240:
                        y = 110
                        y_axis_cnt += 1
                elif direction == 'backward':
                    y -= 2.5
                    if y < 100:
                        y = 250
                        y_axis_cnt += 1
                elif direction == 'good':
                    step += 1
                elif y_axis_cnt > 4:
                    step = 0
                time.sleep(0.1)
                
            case 4:
                print("### 스텝 4 : [컨베이어] 잡기")
                print("Gripper CLOSE")
                y += 15
                z -= 15
                print(f"COLOR:{color_hand}, R:{red_cnt}, G:{green_cnt}, B:{blue_cnt}")
                time.sleep(1)
                mode = 'gripper_close'
                time.sleep(2)
                step += 1
            
            case 5|10:
                print("### 스텝 5|10 : [중간지점] 중간지점으로 이동")
                big_move('middle')      
                if step == 5:
                    step += 1
                if step == 10:
                    step = 0
                
            case 6|9:
                print("### 스텝 6|9 : [블록무더기] 이동 - color 값에 따라 이동 위치 결정")
                if color_hand == 'red':
                    big_move('group_A_level_4')
                if color_hand == 'green':
                    big_move('group_B_level_4')
                if color_hand == 'blue':
                    big_move('group_C_level_4')
                step += 1
                time.sleep(1)
                
            case 7:
                mode = 'coords'
                print("### 스텝 7 : [블록무더기] z축 조정 - xxx_cnt 값에 따라 z축 조정")
                if color_hand == 'red':
                    z -= z_height(red_cnt)
                if color_hand == 'green':
                    z -= z_height(green_cnt)
                if color_hand == 'blue':
                    z -= z_height(blue_cnt)                                        
                time.sleep(2)
                print("이동 완료...")
                step += 1
                
            case 8:
                print("################### 스텝 8 : [블록무더기] 놓기")
                print("Gripper OPEN")
                mode = 'gripper_open'
                print(f"COLOR:{color_hand}, R:{red_cnt}, G:{green_cnt}, B:{blue_cnt}")
                time.sleep(3)
                step += 1

    print("nextstep thread end...")
    

# 코봇 관련
mode = 'angles'
angles = {
    # 'conveyor': [-94.65, -26.8, 68.9, 52.73, -88.68, -9.84],
    # 'conveyor': [-92.54, -28.38, 70.66, 51.66, -88.15, -9.66],
    # 'conveyor': [-98.34, -28.56, 71.54, 52.64, -88.24, -9.84],
    # 'group_A_level_4': [-16.17, 43.15, 23.81, 17.57, -89.73, 0.26],
    # 'group_B_level_4': [3.51, 46.49, 19.24, 17.57, -89.73, 0.26],
    # 'group_C_level_4': [20.21, 57.48, -4.48, 32.78, -90.52, 15.02],
    # 'group_C_level_4': [1.75, -14.5, 98.96, 3.69, -89.2, 0.26],
    # 'group_A_level_4': [-25, 43.15, 23.81, 17.57, -89.73, 0.26],
    # 'group_B_level_4': [-5, 43.15, 23.81, 17.57, -89.73, 0.26],
    # 'group_C_level_4': [15, 43.15, 23.81, 17.57, -89.73, 0.26],
    # 'group_C_level_4': [20.21, 57.48, -4.48, 32.78, -90.52, 15.02],
    # 'group_A_level_4': [-16.17, 43.15, 23.81, 17.57, -89.73, 0.26],
    # 'group_B_level_4': [3.51, 46.49, 19.24, 17.57, -89.73, 0.26],
    # 'group_C_level_4': [1.75, -14.5, 98.96, 3.69, -89.2, 0.26],
    'default': [0, 0, 0, 0, 0, 0],
    'conveyor': [-94.04, -26.36, 69.78, 43.31, -88.68, -6.01],
    'middle': [-48.5, -4.92, 45.08, 26.36, -89.03, -11.07],    
    'group_A_level_4': [-20, 43.15, 23.81, 17.57, -89.73, 0.26],
    'group_B_level_4': [-2, 43.15, 23.81, 17.57, -89.73, 0.26],
    'group_C_level_4': [17, 43.15, 23.81, 17.57, -89.73, 0.26],
}
angles_target = 'middle'
x,y,z,rx,ry,rz = mc.get_coords()

def cobot_thread(mc):
    global mode, x,y,z,rx,ry,rz, angles, angles_target, is_quit

    while True:
        print(mode)
        match mode:
            case 'standby':
                time.sleep(0.1)
            case 'angles':
                mc.sync_send_angles(angles[angles_target], speed=50, timeout=0.5)
                x,y,z,rx,ry,rz = mc.get_coords()
                print('TARGET ANGLES: ', angles_target)
                time.sleep(0.5)
            case 'coords':
                mc.sync_send_coords([x,y,z,rx,ry,rz], speed=50, mode=1, timeout=0.1)
                # print("TARGET COORDS", x,y,z,rx,ry,rz)
                time.sleep(0.1)
            case 'gripper_open':
                mc.set_gripper_value(100, 30)
                time.sleep(0.5)
                mode = 'standby'
            case 'gripper_close':
                mc.set_gripper_value(10, 30)
                time.sleep(0.5)
                mode = 'standby'
                        
        if is_quit:
            break
    
    print("cobot thread end...")


t1 = threading.Thread(target=cobot_thread, daemon=True, args=(mc,))
t2 = threading.Thread(target=nextstep_thread, daemon=True)
t3 = threading.Thread(target=camera_thread, daemon=True, args=(cap,))
t1.start()
t2.start()
t3.start()
t3.join()

cap.release()
cv2.destroyAllWindows()


print("end...")