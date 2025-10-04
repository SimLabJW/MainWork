using System;
using UnityEngine;

public class SLAMAgentManager 
{
    public Coroutine scanCoroutine;

    public Action StartLidar;
    public void LidarStartEvent()
    {
        StartLidar?.Invoke();
    }

    public Action StopLidar;
    public void LidarStopEvent()
    {
        StopLidar?.Invoke();
    }

    public float poseX_m = 0f;       // Δx (m) - 초기화 추가
    public float poseY_m = 0f;       // Δy (m) - 초기화 추가
    public float poseTheta_rad = 0f; // Δθ (rad) - 초기화 추가
}
