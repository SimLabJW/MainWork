# import zmq
# import signal
# import sys

# def signal_handler(sig, frame):
#     print("\n서버를 종료합니다.")
#     socket.close()
#     context.term()
#     sys.exit(0)

# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind("tcp://*:8788")

# signal.signal(signal.SIGINT, signal_handler)

# print("REP 서버 시작됨 (Ctrl+C로 종료)")

# while True:
#     try:
#         message = socket.recv_string()
#         print(f"Unity에서 받은 데이터: {message}")

#         # 예시: RL 제어 명령 반환
#         reply = "action: move_forward"
#         socket.send_string(reply)
#     except zmq.error.ContextTerminated:
#         break
import zmq
import signal
import sys
import matplotlib.pyplot as plt
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:8788")

def signal_handler(sig, frame):
    print("\n서버를 종료합니다.")
    socket.close()
    context.term()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("REP 서버 시작됨 (Ctrl+C로 종료)")

# --- matplotlib 초기화 ---
plt.ion()
fig = plt.figure()
ax = plt.subplot(111, projection="polar")
scat = ax.scatter([], [], s=5, c=[], cmap=plt.cm.viridis, lw=0)
ax.set_rmax(120000)  # 최대 거리 (12m = 12000mm)
ax.grid(True)

while True:
    try:
        message = socket.recv_string()
        socket.send_string("ack")  # Unity에 응답 (필수: REP는 REQ와 쌍이 맞아야 함)

        # 데이터 파싱
        lines = message.strip().splitlines()
        angles = []
        distances = []
        intensities = []

        for line in lines:
            try:
                angle, dist, inten = line.split(",")
                angles.append(float(angle))        # rad
                distances.append(float(dist))      # mm
                intensities.append(float(inten))   # intensity
            except ValueError:
                continue

        # numpy 배열 변환
        angles = np.array(angles)
        distances = np.array(distances)
        intensities = np.array(intensities)

        # 시각화 업데이트
        scat.set_offsets(np.c_[angles, distances])
        scat.set_array(intensities)
        ax.set_title("Unity LiDAR Simulation")

        plt.pause(0.01)

    except zmq.error.ContextTerminated:
        break
