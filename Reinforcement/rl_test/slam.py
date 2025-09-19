import pygame
import numpy as np
import cv2
import math
from utils import raycast_lidar, update_occupancy

# === 설정 ===
MAP_PATH = "map.png"
ROBOT_RADIUS = 5
SPEED = 3
TURN_SPEED = 0.1
LIDAR_BEAMS = 36
LIDAR_RANGE = 80
WIN_SIZE = (800, 600)

def main():
    # 맵 로드
    img = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, WIN_SIZE)
    map_array = (img < 128).astype(np.uint8)  # 1=벽, 0=빈공간
    occ_map = np.zeros_like(map_array)        # SLAM 맵 (0=미탐사,1=빈공간,2=벽)

    pygame.init()
    screen = pygame.display.set_mode(WIN_SIZE)
    clock = pygame.time.Clock()

    # 로봇 초기 상태
    robot_pos = [100, 100]
    robot_angle = 0.0

    running = True
    while running:
        screen.fill((255, 255, 255))

        # 이벤트 처리 (키보드 입력)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            robot_pos[0] += SPEED * math.cos(robot_angle)
            robot_pos[1] += SPEED * math.sin(robot_angle)
        if keys[pygame.K_s]:
            robot_pos[0] -= SPEED * math.cos(robot_angle)
            robot_pos[1] -= SPEED * math.sin(robot_angle)
        if keys[pygame.K_a]:
            robot_angle -= TURN_SPEED
        if keys[pygame.K_d]:
            robot_angle += TURN_SPEED

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # LiDAR 스캔
        scans = raycast_lidar(map_array, robot_pos, robot_angle, LIDAR_BEAMS, LIDAR_RANGE)
        occ_map = update_occupancy(occ_map, robot_pos, scans, robot_angle)

        # === 시각화 ===
        # Ground truth 맵
        surf = pygame.surfarray.make_surface(255 - map_array * 255)
        screen.blit(surf, (0, 0))

        # SLAM 맵
        slam_vis = np.zeros((*occ_map.shape, 3), dtype=np.uint8)
        slam_vis[occ_map == 1] = [200, 200, 200]  # 빈공간
        slam_vis[occ_map == 2] = [0, 0, 0]        # 벽
        slam_surf = pygame.surfarray.make_surface(np.transpose(slam_vis, (1, 0, 2)))
        screen.blit(slam_surf, (0, 0))

        # 로봇
        pygame.draw.circle(screen, (0, 0, 255), (int(robot_pos[0]), int(robot_pos[1])), ROBOT_RADIUS)
        heading_x = int(robot_pos[0] + 15 * math.cos(robot_angle))
        heading_y = int(robot_pos[1] + 15 * math.sin(robot_angle))
        pygame.draw.line(screen, (255, 0, 0), robot_pos, (heading_x, heading_y), 2)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    cv2.imwrite("slam_result.png", (occ_map*120).astype(np.uint8))
    print("SLAM 맵 저장 완료: slam_result.png")

if __name__ == "__main__":
    main()
