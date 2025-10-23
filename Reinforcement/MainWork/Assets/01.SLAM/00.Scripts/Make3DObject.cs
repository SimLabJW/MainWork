using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Make3DObject : MonoBehaviour
{

    void Start()
    {
        GameManager.s_map.PassOnInform -= PickObject;
        GameManager.s_map.PassOnInform += PickObject;
    }

    void PickObject(GameObject Object)
    {
        //  position set
        
    }
}
