# DigitalTwin
1. Remote Control : Unity VR
2. Remote Control : Hand Pose
3. Virtual Envrionment : Robomaster, SLAM
4. Virtual Remote Control
5. New

## 1. Remote Control / Unity VR
Device : Mirobot, Turtlebot, Robomaster(추후)

**Mirobot, Turtlebot**

<!-- ![VR_RemoteControl](./png/VR_RemoteControl.png) -->
<img src="./png/VR_RemoteControl.png" alt="VR_RemoteControl" width="500" height="250"/>

![VR_RemoteControl](./png/turtlebot.gif)

---

## 2. Remote Control / Hand Control

손의 모양, 블럭의 위치 등에 따른 자동차 동작

<p align="center">
  <img src="./png/Cube_RemoteControl1.png" alt="Cube_RemoteControl1" width="250" height="250"/>
  <img src="./png/Cube_RemoteControl2.png" alt="Cube_RemoteControl2" width="250" height="250"/>
  <img src="./png/Cube_RemoteControl3.png" alt="Cube_RemoteControl3" width="250" height="200"/>
</p>

<!-- 위와 같이 <p align="center">로 감싸면 사진이 가로로 나란히 정렬됩니다. 마크다운에서는 이미지를 연속으로 나열하면 자동으로 가로로 배치되지만, 일부 뷰어에서는 세로로 보일 수 있습니다. 위 방법을 사용해보세요. -->

---

## 5. New
### Raspberry Pi 4 Remote Control from Windows

This guide explains how to set up **xrdp** on Raspberry Pi 4 to allow remote desktop access from Windows.  

---

#### 1. Raspberry Pi Setup

1. Install **xrdp** on your Raspberry Pi:

```bash
sudo apt-get update##
sudo apt-get install -y xrdp
```

#### 2. Window Setup

1. Open Remote Desktop Connection
2. If Black Screen Issue
(raspberry pi)

```bash
sudo vi /etc/xrdp/startwm.sh


unset DBUS_SESSION_BUS_ADDRESS
unset XDG_RUNTIME_DIR

test -x /etc/X11/Xsession && exec /etc/X11/Xsession
exec /bin/sh /etc/X11/Xsession

:wq
```