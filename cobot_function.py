import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pymycobot.mycobot import MyCobot
import time
import os



def init(mc):
    mc.send_angles([0,0,0,0,0,0],40)
    time.sleep(4)
    
def zero(mc):
    mc.send_angles([-90,0,-40,-50,90,4],40)
    time.sleep(2)
    
def gripper_open(mc):
    print("OPEN")
    mc.set_eletric_gripper(0)   #그리퍼 현 상태를 닫았다고 인식
    mc.set_gripper_value(100,30)    
    time.sleep(2)

def gripper_close(mc):
    print("CLOSE")
    mc.set_eletric_gripper(1)
    mc.set_gripper_value(16,30) #16까지 닫기
    time.sleep(2)
    
def go_belt_up(mc):
    mc.send_angles([-90,0,-40,-30,93,4],40)
    time.sleep(2)
    
def go_belt_down(mc):
    mc.send_angles([-90,-19,-49,-28,93,4],40)
    time.sleep(2)
    
def middle_point(mc):
    mc.send_angles([0,0,-40,-30,90,4],40)
    time.sleep(2)
    
#block_orange
def bring_orange_three(mc):
    mc.send_angles([80,0,-40,-30,90,4],40)  #orange 색상으로 이동
    time.sleep(4)
    mc.send_angles([80,-39,-47,3,90,10],40) #고개 down
    time.sleep(4)
    gripper_close(mc)
    mc.send_angles([80,0,-40,-30,90,4],40)  #3층의 블록을 잡아서 up
    time.sleep(4)
    middle_point(mc)
    go_belt_up(mc)    #인식장소로 이동
    gripper_open(mc)
    
def bring_orange_two(mc):
    mc.send_angles([80,0,-40,-30,90,4],40)  #orange 색상으로 이동
    time.sleep(4)
    mc.send_angles([80,-43,-54,12,90,5],40) #고개 down
    time.sleep(4)
    gripper_close(mc)
    mc.send_angles([80,0,-40,-30,90,4],40)  #2층의 블록을 잡아서 up
    time.sleep(4)
    middle_point(mc)
    go_belt_up(mc)    #인식장소로 이동
    gripper_open(mc)
    
def bring_orange_one(mc):
    mc.send_angles([80,0,-40,-30,90,4],40)  #orange 색상으로 이동
    time.sleep(4)
    mc.send_angles([80,-48,-53,15,90,0],40) #고개 down
    time.sleep(4)
    gripper_close(mc)
    mc.send_angles([80,0,-40,-30,90,4],40)  #1층의 블록을 잡아서 up
    time.sleep(4)
    middle_point(mc)
    go_belt_up(mc)    #인식장소로 이동
    gripper_open(mc)


#block_green
def bring_green_three(mc):
    mc.send_angles([100,0,-40,-30,90,4],40) #green 색상으로 이동
    time.sleep(2)
    mc.send_angles([100,-38,-54,10,90,8],40)    #고개 down
    time.sleep(2)
    gripper_close(mc)
    mc.send_angles([100,0,-40,-30,90,4],40) #3층의 블록을 잡아서 up
    time.sleep(2)
    middle_point(mc)
    go_belt_up(mc)
    gripper_open(mc)
    
def bring_green_two(mc):
    mc.send_angles([100,0,-40,-30,90,4],40) #green 색상으로 이동
    time.sleep(2)
    mc.send_angles([100,-43,-56,16,90,5],40)    #고개 down
    time.sleep(2)
    gripper_close(mc)
    mc.send_angles([100,0,-40,-30,90,4],40) #2층의 블록을 잡아서 up
    time.sleep(2)
    middle_point(mc)
    go_belt_up(mc)
    gripper_open(mc)
    
def bring_green_one(mc):
    mc.send_angles([100,0,-40,-30,90,4],40) #green 색상으로 이동
    time.sleep(2)
    mc.send_angles([100,-50,-52,16,90,0],40)    #고개 down
    time.sleep(2)
    gripper_close(mc)
    mc.send_angles([100,0,-40,-30,90,4],40) #1층의 블록을 잡아서 up
    time.sleep(2)
    middle_point(mc)
    go_belt_up(mc)
    gripper_open(mc)
    
    
#block_yellow
def bring_yellow_three(mc):
    mc.send_angles([120,0,-40,-30,90,4],40) #yellow 색상으로 이동
    time.sleep(2)
    mc.send_angles([120,-38,-52,6,90,2],40)    #고개 down
    time.sleep(2)
    gripper_close(mc)
    mc.send_angles([120,0,-40,-30,90,4],40) #3층의 블록을 잡아서 up
    time.sleep(2)
    middle_point(mc)
    go_belt_up(mc)
    gripper_open(mc)
    
def bring_yellow_two(mc):
    mc.send_angles([120,0,-40,-30,90,4],40) #yellow 색상으로 이동
    time.sleep(2)
    mc.send_angles([120,-43,-52,10,90,0],40)    #고개 down
    time.sleep(2)
    gripper_close(mc)
    mc.send_angles([120,0,-40,-30,90,4],40) #2층의 블록을 잡아서 up
    time.sleep(2)
    middle_point(mc)
    go_belt_up(mc)
    gripper_open(mc)
    
def bring_yellow_one(mc):
    mc.send_angles([120,0,-40,-30,90,4],40) #yellow 색상으로 이동
    time.sleep(2)
    mc.send_angles([120,-48,-53,15,90,0],40)    #고개 down
    time.sleep(2)
    gripper_close(mc)
    mc.send_angles([120,0,-40,-30,90,4],40) #1층의 블록을 잡아서 up
    time.sleep(2)
    middle_point(mc)
    go_belt_up(mc)
    gripper_open(mc)
    
def operate_orange(mc):
    global i
    gripper_open(mc)
    
    if i==0:
        
        init(mc)
        bring_orange_three(mc)
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero(mc)
        print("first_orange_operation_clear")

    elif i==1:
        init(mc)
        bring_orange_two(mc)
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero(mc)
        print("second_orange_operation_clear")

    elif i==2:
        
        init(mc)
        bring_orange_one(mc)
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero(mc)
        print("third_orange_operation_clear")
        print("All operations clear")

    else :
        print("warning_it_isn't_pyellowictable")
        go_belt_up(mc)
        
def operate_yellow(mc):
    global x
    gripper_open(mc)
    
    if x==0:
        
        init(mc)
        bring_yellow_three(mc)
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero(mc)
        print("first_yellow_operation_clear")

    elif x==1:
        
        init(mc)
        bring_yellow_two(mc)
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero(mc)
        print("second_yellow_operation_clear")

    elif x==2:
    
        init(mc)
        bring_yellow_one(mc)
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero(mc)
        print("third_yellow_operation_clear")
        print("All operations clear")

    else :
        print("warning_it_isn't_pyellowictable")
        go_belt_up(mc)       

def operate_green(mc):
    global w
    gripper_open(mc)
    
    if w==0:
        
        init(mc)
        bring_green_three(mc)
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero(mc)
        print("first_green_operation_clear")

    elif w==1:
        
        init(mc)
        bring_green_two(mc)
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero(mc)
        print("second_green_operation_clear")

    elif w==2:
       
        init(mc)
        bring_green_one(mc)
        # go_stack_first_point_orange_two()
        # go_stack_first_point_orange_three()
        zero(mc)
        print("third_green_operation_clear")
        print("All operations clear")

    else :
        print("warning_it_isn't_pyellowictable")
        go_belt_up(mc)