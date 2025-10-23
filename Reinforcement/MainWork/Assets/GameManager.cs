using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    static GameManager instance;
    public static GameManager Instance { get { Init(); return instance; } }

    SLAMCommManager slam_commmanager = new SLAMCommManager();
    public static SLAMCommManager s_comm { get { return Instance.slam_commmanager; } }

    SLAMAgentManager slam_agent = new SLAMAgentManager();
    public static SLAMAgentManager s_agent { get { return Instance.slam_agent; } }

    SLAMMapManager slam_map = new SLAMMapManager();
    public static SLAMMapManager s_map { get { return Instance.slam_map; } }

    private void Awake() 
    {
        Init();

    }

    static void Init()
    {
        if (instance == null)
        {
            GameObject go = GameObject.Find("GameManager");
            if (go == null)
            {
                go = new GameObject { name = "GameManager" };
                go.AddComponent<GameManager>();

            }
            DontDestroyOnLoad(go);
            instance = go.GetComponent<GameManager>();
        }
    }

    private void OnApplicationQuit()
    {
        s_comm.ShutdownSafely();
    }
}
