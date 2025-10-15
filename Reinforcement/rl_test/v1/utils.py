import numpy as np
import math

def raycast_lidar(map_array, robot_pos, robot_angle, n_beams=36, max_range=100):
    """
    단순 레이캐스트 기반 LiDAR 시뮬레이션
    map_array: numpy 2D 배열 (0=빈칸, 1=벽)
    robot_pos: (x,y)
    robot_angle: 로봇 방향 (라디안)
    n_beams: 빔 개수
    max_range: 최대 거리 (픽셀)
    """
    h, w = map_array.shape
    angles = np.linspace(-math.pi, math.pi, n_beams, endpoint=False)
    scans = []

    for a in angles:
        beam_angle = robot_angle + a
        for r in range(1, max_range):
            x = int(robot_pos[0] + r * math.cos(beam_angle))
            y = int(robot_pos[1] + r * math.sin(beam_angle))
            if x < 0 or y < 0 or x >= w or y >= h:
                scans.append(r)
                break
            if map_array[y, x] == 1:  # 벽 충돌
                scans.append(r)
                break
        else:
            scans.append(max_range)
    return np.array(scans)


def update_occupancy(occ_map, robot_pos, scans, robot_angle):
    """
    Occupancy grid 업데이트
    occ_map: (0=미탐사, 1=빈공간, 2=벽)
    scans: LiDAR 거리값
    """
    h, w = occ_map.shape
    angles = np.linspace(-math.pi, math.pi, len(scans), endpoint=False)

    for i, dist in enumerate(scans):
        beam_angle = robot_angle + angles[i]
        # 빔 내 free space 업데이트
        for r in range(1, dist):
            x = int(robot_pos[0] + r * math.cos(beam_angle))
            y = int(robot_pos[1] + r * math.sin(beam_angle))
            if 0 <= x < w and 0 <= y < h:
                occ_map[y, x] = 1
        # 벽 위치 업데이트
        if dist < max(scans):
            x = int(robot_pos[0] + dist * math.cos(beam_angle))
            y = int(robot_pos[1] + dist * math.sin(beam_angle))
            if 0 <= x < w and 0 <= y < h:
                occ_map[y, x] = 2
    return occ_map
