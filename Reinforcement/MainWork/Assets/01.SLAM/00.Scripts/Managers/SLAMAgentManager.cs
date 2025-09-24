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

    public float deltaX_m;       // Δx (m)
    public float deltaY_m;       // Δy (m)
    public float deltaTheta_rad; // Δθ (rad)
}
