using System;
using UnityEngine;
using System.Collections.Generic;

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

    public bool NextAction = true;

    
    public Action<List<Vector3>> MoveUpdateAgent;
    public void MoveUpdateEvent(List<Vector3> waypoints)
    {
        MoveUpdateAgent?.Invoke(waypoints);
    }

    public string ProcessState = "PROCESS";
    public string RenewalState = "RENEWAL";

    public string AgentState = "PROCESS";


    public Action<List<Vector3>> MoveUpdateCopyAgent;
    public void MoveUpdateCopyEvent(List<Vector3> waypoints)
    {
        MoveUpdateCopyAgent?.Invoke(waypoints);
    }
}
