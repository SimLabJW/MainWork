# rotation_test.py

import json
import time
import math
import sys
import tty
import termios
import zmq

from function import Function_control

SERVER_IP = "192.168.50.75"
SERVER_PORT = 8788

class RotationTest:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{SERVER_IP}:{SERVER_PORT}")
        self.socket.RCVTIMEO = 10000
        self.socket.SNDTIMEO = 10000
        
        self._f = Function_control()
        self.rotation_angle = 30.0

    def send_request(self, data: str):
        try:
            self.socket.send_string(data)
            reply = self.socket.recv_string()
            return json.loads(reply)
        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def get_key(self):
        """키보드 입력 읽기 (기존 방식)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # ✅ 수정
        return ch

    def find_reference_object(self, lidar_buffer):
        """
        특정 거리 범위(0.5~1.5m)에서 가장 많은 점들이 모여있는 각도 찾기
        """
        angle_histogram = {}  # 각도별 점 개수
        
        for line in lidar_buffer:
            parts = line.split(',')
            angle_rad = float(parts[0])
            distance = float(parts[1])
            
            # 0.5~1.5m 범위의 물체만
            if 500 < distance < 1500:
                angle_deg = int(math.degrees(angle_rad))  # 1도 단위로 반올림
                angle_histogram[angle_deg] = angle_histogram.get(angle_deg, 0) + 1
        
        if not angle_histogram:
            return None, 0
        
        # 가장 많은 점이 있는 각도
        max_angle = max(angle_histogram, key=angle_histogram.get)
        max_count = angle_histogram[max_angle]
        
        return max_angle, max_count

    def run(self):
        print("=" * 80)
        print("제자리 회전 테스트 - 라이다 각도 보정 검증")
        print("=" * 80)
        print("\n로봇 앞/옆에 명확한 물체를 배치하세요 (0.5~1.5m 거리)")
        input("준비되면 Enter...")
        
        # 초기 스캔
        print("\n[INIT] 초기 스캔 중...")
        self.send_request(self._f._make_slamData("PROCESS"))
        
        rotation_history = []  # (로봇각도, 물체각도) 기록
        
        while True:
            print("\n" + "=" * 80)
            print(f"[로봇] 현재 방향: {self._f.robot.cur_theta:.1f}°")
            
            # 라이다에서 기준 물체 찾기
            lidar_buffer = self._f.lidar.returning_LidarData()
            obj_angle, obj_count = self.find_reference_object(lidar_buffer)
            
            if obj_angle is not None:
                print(f"[물체] 감지 각도: {obj_angle}° (점 개수: {obj_count})")
                rotation_history.append((self._f.robot.cur_theta, obj_angle))
                
                # 변화량 계산
                if len(rotation_history) >= 2:
                    prev_robot, prev_obj = rotation_history[-2]
                    curr_robot, curr_obj = rotation_history[-1]
                    
                    robot_change = curr_robot - prev_robot
                    obj_change = curr_obj - prev_obj
                    
                    # 각도 정규화 (-180 ~ 180)
                    if obj_change > 180:
                        obj_change -= 360
                    elif obj_change < -180:
                        obj_change += 360
                    
                    print(f"[변화] 로봇: {robot_change:+.1f}°, 물체: {obj_change:+.1f}°")
                    
                    if abs(robot_change - obj_change) < 5:
                        print("       ✅ 보정 정상 (차이 < 5°)")
                    else:
                        print(f"       ❌ 보정 오류 (차이: {abs(robot_change - obj_change):.1f}°)")
            else:
                print("[물체] 0.5~1.5m 범위에서 물체를 찾을 수 없음")
            
            print("=" * 80)
            print("[입력] q:좌회전 | e:우회전 | r:기록초기화 | x:종료")
            
            key = self.get_key()
            print(key)
            
            if key == 'x':
                break
            elif key == 'r':
                rotation_history = []
                print("\n기록 초기화됨")
            elif key == 'q':
                print(f"\n[회전] 좌회전 {self.rotation_angle}°")
                self._f.robot._control_robot_motion((0.0, -self.rotation_angle))
                time.sleep(0.5)
                self._f.lidar._scan_360()
                self.send_request(self._f._make_slamData("RENEWAL"))
            elif key == 'e':
                print(f"\n[회전] 우회전 {self.rotation_angle}°")
                self._f.robot._control_robot_motion((0.0, self.rotation_angle))
                time.sleep(0.5)
                self._f.lidar._scan_360()
                self.send_request(self._f._make_slamData("RENEWAL"))
        
        print("\n[결과 요약]")
        for i, (robot, obj) in enumerate(rotation_history):
            print(f"  {i}: 로봇={robot:.1f}°, 물체={obj}°")
        
        self.socket.close()
        self.context.term()


if __name__ == "__main__":
    test = RotationTest()
    test.run()