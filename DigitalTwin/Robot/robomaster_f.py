import sys
import time
import math
from typing import List, Tuple, Callable, Optional
from collections import deque

import robomasterpy as rm

class RobomasterControl:
    def __init__(self, robot_ip: str = ""):
        self.robot = None

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_delta_x = 0.0
        self.pose_delta_y = 0.0

        self.cur_theta = 0.0  # 초기 방향: y축 위 (북쪽)
        self.pre_theta = 0.0
        self.pose_delta_theta = 0.0

        self.rotation = 0
        self.speed = 0

    def _connect_robot(self):
        try:
            self.robot = rm.Commander(timeout=15.0)
            ver = self.robot.version()
            print(f"[ROBOT] Version: {ver}")
            print(f"[ROBOT] Initial heading: {self.cur_theta}° (y+ direction)")
        except Exception as e:
            print(f"[ERROR] 로봇 연결 실패: {e}")
            sys.exit(1)

    def _stop_robot_motion(self):
        self.robot.chassis_move(x=0, y=0, z=0)

    def _control_robot_motion(self, cmd):
        distance, rotation = cmd
        self.robot_chassis_move(distance, rotation)
        self._update_pose()

    def robot_chassis_move(self, distance, rotation):
        # ===== 1. 회전 ===== #
        if rotation != 0:
            self.robot.chassis_move(x=0, y=0, z=rotation)
            print(f"[MOVE] Rotating {rotation}°...")
            time.sleep(1.5)
            self.robot.chassis_move(x=0, y=0, z=0)
        self.cur_theta += rotation
        print(f"[MOVE] Now facing {self.cur_theta}°")
        
        time.sleep(0.3)
        
        # ===== 2. 직진 ===== #
        if distance > 0:
            self.robot.chassis_move(x=distance, y=0, z=0)
            print(f"[MOVE] Moving forward {distance}m...")
            time.sleep(1.5)
            self.robot.chassis_move(x=0, y=0, z=0)
    
        # ===== 3. 오도메트리 계산 ===== #
        theta_rad = math.radians(self.cur_theta)
        
        # ✅ 여기에 디버그 추가
        print(f"[DEBUG] distance={distance}, cur_theta={self.cur_theta}°")
        print(f"[DEBUG] theta_rad={theta_rad:.4f} rad")
        print(f"[DEBUG] cos(theta)={math.cos(theta_rad):.4f}, sin(theta)={math.sin(theta_rad):.4f}")
        
        self.pose_x = distance * math.cos(theta_rad)
        self.pose_y = distance * math.sin(theta_rad)
        
        print(f"[ODOM] Delta: x={self.pose_x:.3f}, y={self.pose_y:.3f}, theta={rotation}°")

    def _update_pose(self):
        self.pose_delta_x += self.pose_x 
        self.pose_delta_y += self.pose_y 

        self.pose_delta_theta = math.radians(self.cur_theta)
        self.pre_theta = self.cur_theta
        
        print(f"[POSE] Accumulated: x={self.pose_delta_x:.3f}, y={self.pose_delta_y:.3f}")

    def returning_pose(self):
        return self.pose_delta_x, self.pose_delta_y, self.pose_delta_theta