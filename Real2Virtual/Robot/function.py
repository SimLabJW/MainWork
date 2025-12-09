from robomaster_f import RobomasterControl
from lidar_f import *

# ============ Lidar 설정 ============== #
LIDAR_PORT = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
LIDAR_BAUD = 115200

# ============ Robot 설정 ============== #
ROBOT_IP = ""


class Function_control:
    def __init__(self):
        # ====== Init Robot & Lidar ========== #
        self.robot = RobomasterControl(ROBOT_IP)
        self.lidar = LidarController(LIDAR_PORT, LIDAR_BAUD)    

        # ======== Connect Robot & LiDAR ======== #
        self._connect_lidar()
        self._connect_robomaster()

        # ======== path parameter ======= #
        self.path_cnt = 0

    def _connect_lidar(self):
        self.lidar._connect_lidar()
        time.sleep(1)
        # 초기 스캔 시 로봇 방향 전달
        self.lidar._scan_360()

    def _connect_robomaster(self):
        self.robot._connect_robot()

    def _calculate_motion(self, path):
        if len(path) != 0:
            self.robot._control_robot_motion(path[self.path_cnt])
            self.path_cnt += 1

        # 스캔 시 현재 로봇 방향 전달
        self.lidar._scan_360()

        return self.path_cnt

    def _calculate_path2motion(self, path, min_rotation=5.0, max_rotation=30.0, 
                          min_distance=0.15, max_distance=0.5):

        if len(path) < 2:
            return []
        
        # ===== 1. 큰 방향 전환 지점만 찾기 (임계값 증가) ===== #
        segments = []
        segment_start = 0
        angle_threshold = 15.0  # ✅ 5도 → 15도 (작은 각도 변화는 무시)
        
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_point = path[i + 1]
            
            dx1 = curr[0] - prev[0]
            dy1 = curr[1] - prev[1]
            dx2 = next_point[0] - curr[0]
            dy2 = next_point[1] - curr[1]
            
            dist1 = math.sqrt(dx1**2 + dy1**2)
            dist2 = math.sqrt(dx2**2 + dy2**2)
            
            if dist1 < 0.001 or dist2 < 0.001:
                continue
            
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            angle_diff = abs(angle2 - angle1)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            # ✅ 15도 이상 차이날 때만 세그먼트 분할
            if angle_diff > math.radians(angle_threshold):
                segments.append((segment_start, i))
                segment_start = i
        
        segments.append((segment_start, len(path) - 1))
        
        
        # ===== 2. 각 구간을 (회전, 직진) 명령으로 변환 ===== #
        moving_path = []
        current_angle = self.robot.cur_theta
        
        for seg_idx, (start_idx, end_idx) in enumerate(segments):
            start_point = path[start_idx]
            end_point = path[end_idx]
            
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            segment_distance = math.sqrt(dx**2 + dy**2)
            segment_angle = math.degrees(math.atan2(dy, dx))
            
            if segment_distance < 0.01:
                continue
            
            # 필요한 회전 계산
            rotation_needed = segment_angle - current_angle
            while rotation_needed > 180:
                rotation_needed -= 360
            while rotation_needed < -180:
                rotation_needed += 360
            
            # ===== 회전 명령 (유연한 분할: 5~30도) ===== #
            if abs(rotation_needed) > min_rotation:
                remaining_rot = rotation_needed
                while abs(remaining_rot) > min_rotation:
                    # ✅ 남은 각도가 작으면 한번에, 크면 최대치로
                    if abs(remaining_rot) <= max_rotation:
                        # 한번에 회전 가능
                        moving_path.append((0.0, remaining_rot))
                        current_angle += remaining_rot
                        remaining_rot = 0
                    else:
                        # max_rotation만큼 회전
                        rot_step = math.copysign(max_rotation, remaining_rot)
                        moving_path.append((0.0, rot_step))
                        remaining_rot -= rot_step
                        current_angle += rot_step
            
            # ===== 직진 명령 (유연한 분할: 0.15~0.5m) ===== #
            remaining_dist = segment_distance
            while remaining_dist > 0.01:
                if remaining_dist <= max_distance:
                    # ✅ 남은 거리가 max 이하면 한번에
                    if remaining_dist >= min_distance:
                        moving_path.append((remaining_dist, 0.0))
                    else:
                        # 너무 짧으면 min_distance로 (약간 오버슛)
                        moving_path.append((min_distance, 0.0))
                    remaining_dist = 0
                else:
                    # max_distance만큼 이동
                    moving_path.append((max_distance, 0.0))
                    remaining_dist -= max_distance
        
        # ===== 명령 요약 출력 ===== #
        rot_count = sum(1 for cmd in moving_path if cmd[1] != 0)
        move_count = sum(1 for cmd in moving_path if cmd[0] != 0)
        print(f"[PATH] Rotation commands: {rot_count}, Move commands: {move_count}")
        
        return moving_path


    def _make_slamData(self, state):
        self.robot._stop_robot_motion()
        pose_x, pose_y, pose_theta = self.robot.returning_pose()
        
        lidar_data = self.lidar.returning_LidarData()
        
        lines = lidar_data.copy()
        lines.append(f"POSE,{pose_x:.4f},{pose_y:.4f},{pose_theta:.4f}")
        lines.append(state)
        
        return "\n".join(lines)