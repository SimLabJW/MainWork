# manual_control.py

import json
import time
import math
import sys
import tty
import termios
from typing import Optional
import zmq

from function import Function_control

# ============ zmq 설정 ============== #
SERVER_IP = "192.168.50.75"
SERVER_PORT = 8788

class ManualControlSLAM:
    def __init__(self) -> None:
        # ============ ZMQ Connect =========== #
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{SERVER_IP}:{SERVER_PORT}")
        self.socket.RCVTIMEO = 10000
        self.socket.SNDTIMEO = 10000
        print(f"[ZMQ] Connected to {SERVER_IP}:{SERVER_PORT}")

        self.running = True
        self.agent_state = "PROCESS"
        
        # ============ SLAM Function =========== #
        self._f = Function_control()
        
        # ============ 수동 제어 파라미터 =========== #
        self.move_distance = 0.5  # 한번에 이동할 거리 (m)
        self.rotation_angle = 30.0  # 한번에 회전할 각도 (degree)

    def send_request(self, data: str) -> Optional[dict]:
        """동기 REQ/REP"""
        try:
            self.socket.send_string(data)
            reply = self.socket.recv_string()
            return json.loads(reply)
        except zmq.Again:
            print("[WARN] ZMQ timeout")
            return None
        except Exception as e:
            print(f"[ERROR] ZMQ: {e}")
            return None

    def get_key(self):
        """키보드 입력 읽기"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def execute_manual_command(self, key):
        """키보드 명령 실행"""
        if key == 'w':
            print(f"\n[MANUAL] 직진 {self.move_distance}m")
            self._f.robot._control_robot_motion((self.move_distance, 0.0))
            time.sleep(0.5)
            self._f.lidar._scan_360()
            return True
            
        elif key == 'q':
            print(f"\n[MANUAL] 좌회전 {self.rotation_angle}°")
            self._f.robot._control_robot_motion((0.0, -self.rotation_angle))
            time.sleep(0.5)
            self._f.lidar._scan_360()
            return True
            
        elif key == 'e':
            print(f"\n[MANUAL] 우회전 {self.rotation_angle}°")
            self._f.robot._control_robot_motion((0.0, self.rotation_angle))
            time.sleep(0.5)
            self._f.lidar._scan_360()
            return True
        
        elif key == 'r':
            print(f"\n[STATE] Switching to PROCESS mode")
            self.agent_state = "PROCESS"
            # 상태만 변경, 라이다 스캔은 안함
            return True  # 상태 출력만 업데이트
            
        elif key == 'x':
            print("\n[MANUAL] 종료")
            return False
            
        else:
            print(f"\n[WARN] Unknown key: {key}")
            return None

    def print_status(self):
        """현재 상태 출력"""
        pose_x, pose_y, pose_theta = self._f.robot.returning_pose()
        print("\n" + "=" * 70)
        print(f"[STATUS] State: {self.agent_state}")
        print(f"[STATUS] Robot Odometry:")
        print(f"         Position: x={pose_x:.4f}m, y={pose_y:.4f}m")
        print(f"         Angle: {self._f.robot.cur_theta:.1f}° (absolute={math.degrees(pose_theta):.1f}°)")
        print("=" * 70)
        print("[CONTROLS] w:직진(0.3m) | q:좌회전(30°) | e:우회전(30°)")
        print("           r:PROCESS모드 | x:종료")
        print("=" * 70)

    def run(self):
        print("=" * 70)
        print("Manual Control SLAM Client - Debugging Mode")
        print("=" * 70)
        print("\n⚠️  로봇을 원하는 방향으로 배치하세요!")
        print("    초기 각도: 0° (로봇이 보고 있는 방향)\n")
        
        try:
            # ===== 1. 초기 PROCESS 데이터 전송 ===== #
            print("\n[INIT] Sending initial PROCESS data...")
            reply_data = self.send_request(self._f._make_slamData(self.agent_state))
            
            if reply_data:
                reply_res = reply_data.get("result", {})
                status = reply_res.get("status", "")
                print(f"[INIT] Server response: {status}")
                
                # path가 있으면 RENEWAL 모드로 전환
                d_path = reply_res.get("path", [])
                if len(d_path) > 0:
                    self.agent_state = "RENEWAL"
                    print(f"[INIT] → Switched to RENEWAL mode (path received: {len(d_path)} points)")
                else:
                    print(f"[INIT] → Staying in PROCESS mode (no path)")
            
            # ===== 2. 수동 제어 루프 ===== #
            self.print_status()
            
            while self.running:
                print("\n입력: ", end='', flush=True)
                key = self.get_key()
                print(key)
                
                result = self.execute_manual_command(key)
                
                if result is False:  # 종료
                    break
                
                elif result is None:  # 상태 변경만 (r 키)
                    self.print_status()
                    
                elif result is True:  # 로봇 이동 명령 (w, q, e)
                    # SLAM 데이터 전송
                    print("\n[SLAM] Sending updated data...")
                    reply_data = self.send_request(self._f._make_slamData(self.agent_state))
                    
                    if reply_data:
                        reply_res = reply_data.get("result", {})
                        status = reply_res.get("status", "")
                        slam_pose = reply_res.get("slam_pose", {})
                        
                        print(f"\n[SLAM] Server response: {status}")
                        
                        if slam_pose:
                            print(f"[SLAM] SLAM Estimated Pose:")
                            print(f"       x={slam_pose.get('x', 0):.4f}m")
                            print(f"       y={slam_pose.get('y', 0):.4f}m")
                            print(f"       theta={slam_pose.get('theta', 0):.4f}rad ({math.degrees(slam_pose.get('theta', 0)):.1f}°)")
                            
                            # 오도메트리와 비교
                            pose_x, pose_y, _ = self._f.robot.returning_pose()
                            diff_x = abs(pose_x - slam_pose.get('x', 0))
                            diff_y = abs(pose_y - slam_pose.get('y', 0))
                            print(f"\n[DIFF] Odometry vs SLAM:")
                            print(f"       Δx = {diff_x:.4f}m")
                            print(f"       Δy = {diff_y:.4f}m")
                        
                        # path가 있으면 RENEWAL 모드로 자동 전환
                        d_path = reply_res.get("path", [])
                        if status == "continue" and len(d_path) > 0:
                            old_state = self.agent_state
                            self.agent_state = "RENEWAL"
                            if old_state != self.agent_state:
                                print(f"\n[AUTO] Switched to RENEWAL mode (path received: {len(d_path)} points)")
                    
                    self.print_status()
                
        except KeyboardInterrupt:
            print("\n\n[MAIN] ⚠️ Interrupted by user")
        
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n" + "=" * 70)
        
        try:
            self.socket.close(0)
            self.context.term()
        except:
            pass
        
        print("[SYSTEM] Shutdown complete")
        print("=" * 70)


if __name__ == "__main__":
    client = ManualControlSLAM()
    client.run()