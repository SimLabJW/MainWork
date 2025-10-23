using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class SLAMMapManager 
{
    //map
    public Action StartMaking;
    public void StartMakingEvent()
    {
        StartMaking?.Invoke();
    }

    public Action<GameObject> PassOnInform;
    public void PassOnInformEvent(GameObject Object)
    {
        PassOnInform?.Invoke(Object);
    }

    // object
    public Vector3 SelectObjectPosition;
    public bool SelectObjectON = false;

    public Action Capturing;
    public void CapturingEvnet()
    {
        Capturing?.Invoke();
    }

}
