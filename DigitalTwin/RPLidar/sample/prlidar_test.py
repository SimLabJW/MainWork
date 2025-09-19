# 이 에러는 RPLidar('/dev/ttyUSB0')에서 지정한 포트('/dev/ttyUSB0')를 찾을 수 없어서 발생합니다.
# Windows 환경에서는 '/dev/ttyUSB0'가 아니라 'COMx' (예: 'COM3')와 같은 포트명을 사용해야 합니다.
# 실제 연결된 Lidar의 포트명을 확인한 후 아래와 같이 수정하세요.

from rplidar import RPLidar

# Windows에서는 아래와 같이 COM 포트명을 사용해야 합니다.
# 예시: lidar = RPLidar('COM3')
lidar = RPLidar('COM3')  # 실제 연결된 포트로 변경하세요

info = lidar.get_info()
print(info)

health = lidar.get_health()
print(health)

for i, scan in enumerate(lidar.iter_scans()):
    print('%d: Got %d measurments' % (i, len(scan)))
    if i > 10:
        break

lidar.stop()
lidar.stop_motor()
lidar.disconnect()