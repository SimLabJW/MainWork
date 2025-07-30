using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using UnityEngine;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour
{
    [Header("Water Object")]
    public GameObject Ocean;

    [Header("Sea Agent Floater")]
    public GameObject Floater;

    static GameManager instance;
    public static GameManager Instance { get { Init(); return instance; } }

    //SceneSwtichingManager _sceneswitchingManager = new SceneSwtichingManager();
    //public static SceneSwtichingManager Scene_sw { get { return Instance._sceneswitchingManager; } }

    SimulationManager _simulationManager = new SimulationManager();
    public static SimulationManager simulation { get { return Instance._simulationManager; } }

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
}
