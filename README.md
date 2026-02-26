# RealToSim Integrated System

An integrated pipeline for **autonomous robot exploration** and **automatic virtual environment reconstruction**, combining DQN-based frontier selection, Graph-SLAM mapping, and Unity 3D visualization into a single unified framework.

## About

Unmanned systems operating in disaster sites or hazardous environments typically rely on pre-built maps or remote human control. **RealToSim** addresses this by automating the entire workflow — from physical exploration to virtual reconstruction — without any prior environmental information.

The system integrates three core components:

- **DQN-based frontier selection** — A Deep Q-Network agent combined with heuristic scoring selects exploration targets efficiently, even in structurally complex environments.
- **Graph-SLAM real-time mapping** — ICP scan matching, loop closure detection, and Gauss-Newton optimization generate a globally consistent occupancy grid map throughout long-duration exploration.
- **Unity 3D environment construction** — The generated map is automatically converted into a 3D virtual space in real time, ready for use in reinforcement learning training, path planning validation, and robot simulation.

---

## Requirements

### Python (server & robot side)

- Python 3.8+
- Stable-Baselines3
- OpenAI Gym
- NumPy, SciPy, OpenCV

Full dependency list:
- Server: `Real2Sim/requirements_slam_rl.txt`
- Robot: `Real2Sim/Robot/requirement_robot.txt`

### Unity (virtual environment)

- Unity **2022.3.19f1**
- High Definition Render Pipeline (HDRP)

### Hardware

- DJI Robomaster EP (robot platform)
- RPLIDAR A1 (LiDAR sensor, max range 12 m, 360°)
- Raspberry Pi 4 (4 GB RAM, onboard robot controller)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/SimLabJW/MainWork.git
cd MainWork
git checkout simulation
```

**2. Install Python dependencies (server)**

```bash
pip install -r Real2Sim/requirements_slam_rl.txt
```

**3. Install Python dependencies (robot)**

```bash
pip install -r Real2Sim/Robot/requirement_robot.txt
```

**4. Open Unity project**

Open `Reinforcement/MainWork/` in **Unity Hub** using Unity version **2022.3.19f1**.
Ensure the HDRP package is installed via the Unity Package Manager.

---

## How to Run

The system consists of three components. Launch them in the following order.

### Step 1 — Launch Unity Environment

Open the Unity project at `Reinforcement/MainWork/` and run the scene from `Assets/Scenes/`.

The Unity application listens for occupancy grid map data streamed from the server and reconstructs the 3D virtual space in real time.

### Step 2 — Start the Exploration Server

```bash
cd "Real2Sim/Exploration(SLAM)"
python server.py
```

The server handles frontier detection, DQN-based frontier selection, Graph-SLAM mapping, and bidirectional communication with both the robot and Unity.

### Step 3 — Start the Robot

```bash
cd Real2Sim/Robot
python main.py
```

The robot module connects to the DJI Robomaster EP via SDK, collects LiDAR scan data, and streams it to the server.

---

## Repository Structure

The source code for this system is distributed across the following directories:

| Folder | Description |
|--------|-------------|
| `Real2Sim/Exploration(SLAM)/` | Main server, frontier detection, DQN agent, Graph-SLAM modules |
| `Real2Sim/Robot/` | Robot interface for DJI Robomaster EP (LiDAR, motion control) |
| `Reinforcement/FrontierRL/` | DQN training environment for frontier selection |
| `Reinforcement/MainWork/Assets/` | Unity project assets (3D virtual environment construction) |

### `Real2Sim/Exploration(SLAM)/`

    Exploration(SLAM)/
    ├── server.py                   # Main server entry point
    ├── frontier/
    │   ├── frontier.py             # Frontier data structure
    │   ├── frontier_wfd.py         # Wavefront Frontier Detection (WFD)
    │   ├── global_planner.py       # A* global path planning
    │   ├── experiment_selector.py  # Baseline selector (MRTSP / Heuristic / DQN)
    │   └── mrtsp_selector.py       # MRTSP baseline selector
    ├── rl/
    │   ├── Frontier_Agent.py       # DQN agent (Stable-Baselines3)
    │   ├── rl_data_logger.py       # Training log recorder
    │   └── models/                 # Saved DQN model weights
    ├── slams/
    │   ├── slam_unity_v.py         # Graph-SLAM for Unity virtual environment
    │   ├── slam_real_v.py          # Graph-SLAM for real robot environment
    │   └── newslam/
    │       ├── frontend.py         # Scan matching (ICP-based)
    │       ├── loop_closure.py     # Loop closure detection (KD-tree)
    │       ├── graph.py            # Pose graph structure
    │       ├── icp.py              # ICP implementation
    │       ├── edge_odometry.py    # Odometry edge definition
    │       ├── vertex.py           # Graph vertex definition
    │       ├── pose_se2.py         # SE(2) pose representation
    │       ├── main_g2o.py         # Gauss-Newton graph optimization
    │       ├── main_clf.py         # Classifier-based loop detection
    │       ├── chi2_grad_hess.py   # Chi-squared gradient/Hessian
    │       ├── load.py             # Data loader
    │       └── util.py             # Utility functions
    └── data_exports/               # Exported map and log data

### `Real2Sim/Robot/`

    Robot/
    ├── main.py                     # Robot main entry point
    ├── robomaster.py               # DJI Robomaster EP SDK interface
    ├── lidar_f.py                  # RPLIDAR A1 data acquisition
    ├── function.py                 # Motion control utilities
    ├── manual_control.py           # Manual teleoperation mode
    └── requirement_robot.txt       # Robot-side Python dependencies

### `Reinforcement/FrontierRL/`

    FrontierRL/
    ├── train_frontier.py           # DQN training script
    └── envs/
        ├── frontier_dqn_env.py     # OpenAI Gym-compatible DQN environment
        └── frontier/               # Shared frontier utilities

### `Reinforcement/MainWork/Assets/` (Unity)

    Assets/
    ├── GameManager.cs              # Main Unity game manager
    ├── GameManager.prefab
    ├── 01.SLAM/
    │   ├── 00.Scripts/             # C# scripts for map reception & mesh generation
    │   └── Resources/              # Unity resources for virtual env rendering
    ├── Scenes/                     # Unity scene files
    ├── Plugins/                    # External Unity plugins
    └── HDRPDefaultResources/       # HDRP rendering resources

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## 주요 작업(Main Work) 설명
- **DEVS 시뮬레이션**: 시스템 동작을 분석/예측하기 위한 Discrete Event Simulation 기반 환경 구현.
  - [DEVS 시뮬레이션](./Simulation(UnityVisual)/README.md)
- **강화학습(Reinforcement Learning)**: 가상 환경에서 다양한 알고리즘을 적용하여 로봇/에이전트 제어.
  - [강화학습 (Reinforcement Learning)](./Reinforcement/README.md)
- **Real2Sim(Env)**: 실제 환경을 바탕으로 한 시뮬레이션 환경 구축(현실-가상 전이).
  - [Real2Sim(Env)](./Real2Sim/README.md)
- **Unity**: 3D 시각화, 인터랙션 및 원격제어 UI/UX 플랫폼.
  - [Unity Use](./UnityUsage/README.md)
