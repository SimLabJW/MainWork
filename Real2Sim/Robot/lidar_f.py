from random import uniform
import sys
import time
import math
from typing import Tuple, List
import numpy as np

from rplidar import RPLidar, RPLidarException

class LidarController:
    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self.lidar = None
        self.port = port
        self.baudrate = baudrate

        # 라이다 장착 각도 offset (degree)
        self.lidar_offset_deg = 180.0

        # Set Parameter
        self.pointPerScan = 1450

        # Lidar Buffer
        self.lidar_buffer = []

    def _connect_lidar(self):
        self.lidar = RPLidar(self.port, baudrate=self.baudrate)

        print("[RPLIDAR] Resetting lidar...")
        self.lidar.stop()
        self.lidar.stop_motor()
        time.sleep(1)
        
        print("[RPLIDAR] Starting motor...")
        self.lidar.start_motor()
        time.sleep(2)
        
        health = self.lidar.get_health()
        print(f"[RPLIDAR] Health status: {health}")
        
        info = self.lidar.get_info()
        print(f"[RPLIDAR] Device info: {info}")
        
        self.scans = self.lidar.iter_scans()
        
        print("[RPLIDAR] Connected successfully!")
        print(f"[RPLIDAR] Lidar offset: {self.lidar_offset_deg}°")

    def _scan_360(self, min_coverage=0.95, max_scans=20):
        """
        360도 커버리지가 충분할 때까지 스캔 수집
        robot_heading_deg: 로봇의 현재 방향 (degree)
        """
        self.lidar_buffer = []
        
        try:
            self.lidar.clean_input()
        except:
            pass

        accumulated_scans = []
        coverage = 0.0  
        scan_count = 0
        retry_count = 0
        max_retries = 3
        
        while coverage < min_coverage and scan_count < max_scans:  
            try:
                scan = next(self.scans)
                accumulated_scans.extend(scan)
                
                coverage = self._get_complete_360(accumulated_scans)
                scan_count += 1
                
                print(f"Scan {scan_count}: {len(scan)} measurements, Coverage: {coverage*100:.1f}%")
                
                retry_count = 0

            except Exception as e:
                retry_count += 1
                
                if retry_count >= max_retries:
                    break
                
                try:
                    self.lidar.stop()
                    self.lidar.stop_motor()
                    self.lidar.clean_input()
                    time.sleep(2)
                    
                    self.lidar.start_motor()
                    time.sleep(2)
                    
                    self.scans = self.lidar.iter_scans()
                    time.sleep(1)
                    
                except Exception as restart_error:
                    time.sleep(1)
        
        print(f"[LIDAR] Collected {len(accumulated_scans)} total measurements")
        uniform_data = self._scanMethod_unity(accumulated_scans, self.pointPerScan)
        self._add_lidar(uniform_data)

    def _scanMethod_unity(self, scan, pointPerScan):
        if len(scan) == 0:
            print("[WARN] No scan data available")
            return []
        
        scan_sorted = sorted(scan, key=lambda x: x[1])
        angle_step = 360.0 / pointPerScan
        result = []
        
        for i in range(pointPerScan):
            target_angle = i * angle_step
            closest = min(scan_sorted, key=lambda x: abs(x[1] - target_angle))
            quality, angle, distance = closest
            
            result.append({
                'actual_angle': target_angle,
                'distance': distance,
                'intensity': quality
            })
        
        return result

    def _get_complete_360(self, accumulated_scans):
        if len(accumulated_scans) == 0:
            return 0.0
        
        num_bins = 72
        bin_size = 360.0 / num_bins
        bins_covered = set()
        
        for quality, angle, distance in accumulated_scans:
            bin_index = int(angle / bin_size) % num_bins
            bins_covered.add(bin_index)
        
        coverage = len(bins_covered) / num_bins
        return coverage

    def _add_lidar(self, lidarData):
        """
        라이다 데이터를 로봇 로컬 좌표계로 (SLAM이 변환함)
        """
        for item in lidarData:
            # 1. 라이다 로컬 각도
            angle_local = item['actual_angle']
            
            # 2. 라이다 장착 offset만 적용 (로봇 로컬)
            angle_robot_local = angle_local + self.lidar_offset_deg
            
            # 3. 0~360 정규화
            while angle_robot_local >= 360:
                angle_robot_local -= 360
            while angle_robot_local < 0:
                angle_robot_local += 360
            
            angle_rad = math.radians(angle_robot_local)
            self.lidar_buffer.append(f"{angle_rad:.4f},{item['distance']:.1f},{item['intensity']:.0f}")
        
        print(f"[LIDAR] Sent in robot local frame")

    def returning_LidarData(self):
        """라이다 버퍼 반환 (이미 월드 좌표계로 변환됨)"""
        return self.lidar_buffer