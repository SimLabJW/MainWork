using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class SLAMMapManager 
{
    public GameObject CopyMap;
    public int[,] GridData;          // 0: Free, 1: Occupied
    public Vector2 Origin;           // origin.x, origin.y
    public float CellSize;           // resolution


    public GameObject CopyAgent;
    
    public Action StartMaking;
    public void StartMakingEvent()
    {
        StartMaking?.Invoke();
    }

    // agent move
    public Action<Vector3, Vector3> PassOnInform;
    public void PassOnInformEvent(Vector3 target, Vector3 robot)
    {
        PassOnInform?.Invoke(target, robot);
    }
    // agent rottation
    public Action<string> RotationCopyAgent;
    public void RotationCopyAgentEvent(string dir)
    {
        RotationCopyAgent?.Invoke(dir);
    }

    // object
    public Vector3 SelectObjectPosition;
    public bool SelectObjectON = false;

    public Action Capturing;
    public void CapturingEvnet()
    {
        Capturing?.Invoke();
    }

    // minimap
    public Action<GameObject> StartMiniMap;
    public void StartMiniMapEvent(GameObject Map)
    {
        StartMiniMap?.Invoke(Map);
    }

    public Action SwithchingMap;
    public void SwithicngMapEvent()
    {
        SwithchingMap?.Invoke();
    }

}
