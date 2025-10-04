import matplotlib.pyplot as plt
import numpy as np

def parse_messages(file_path):
    scan_points = []
    path_points = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    current_pose = (0.0, 0.0, 0.0)  # (X, Y, θ_rad)
    current_scan = []

    for line in lines:
        parts = line.strip().split(",")

        if parts[0] == "POSE":
            # 라이다 점들을 이전 pose 기준으로 변환하여 누적
            theta_rad = current_pose[2]  # 이미 rad
            rot = np.array([
                [np.cos(theta_rad), -np.sin(theta_rad)],
                [np.sin(theta_rad),  np.cos(theta_rad)]
            ])
            for px, py in current_scan:
                # 좌표계 변환 (Unity 기준 → matplotlib 기준: (y, x))
                global_point = np.dot(rot, np.array([px, py])) + np.array([current_pose[1], current_pose[0]])
                scan_points.append(global_point)
            current_scan = []  # 스캔 점 초기화

            # pose 업데이트
            current_pose = tuple(map(float, parts[1:4]))
            path_points.append([current_pose[1], current_pose[0]])  # (y, x) 순서로 경로 저장

        else:
            try:
                angle, distance, intensity = map(float, parts)
                if 0 < distance < 80000:
                    r = distance / 1000.0  # mm → meters
                    px = r * np.cos(angle)
                    py = r * np.sin(angle)
                    current_scan.append((px, py))
            except:
                continue

    return np.array(scan_points), np.array(path_points)


def plot_map(scan_points, path_points):
    plt.figure(figsize=(10, 10))
    plt.title("LiDAR Map with Robot Path")
    plt.axis("equal")
    plt.grid(True)

    if len(scan_points) > 0:
        plt.scatter(scan_points[:, 0], scan_points[:, 1], s=1, c='blue', label="Mapped Points")

    if len(path_points) > 0:
        plt.plot(path_points[:, 0], path_points[:, 1], 'r-', label="Robot Path")
        plt.plot(path_points[-1, 0], path_points[-1, 1], 'ro', label="Current Pose")

    plt.legend()
    plt.show()

# 실행
if __name__ == "__main__":
    scan, path = parse_messages("latest_message.txt")
    plot_map(scan, path)
