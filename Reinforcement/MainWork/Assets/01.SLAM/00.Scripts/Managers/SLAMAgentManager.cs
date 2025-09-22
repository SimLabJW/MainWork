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
}
