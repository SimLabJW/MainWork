using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    [Header("Water Object")]
    public GameObject Ocean;

    [Header("Sea Agent Floater")]
    public GameObject Floater;

    static GameManager instance;
    public static GameManager Instance { get { Init(); return instance; } }

    CommunicationManager _communicationManager = new CommunicationManager();
    public static CommunicationManager communication { get { return Instance._communicationManager; } }

    CreateScenarioManager _createScenarioManager = new CreateScenarioManager();
    public static CreateScenarioManager createScenario { get { return Instance._createScenarioManager; } }

    ScenarioEditManager _scenarioEditManager = new ScenarioEditManager();
    public static ScenarioEditManager scenarioEdit { get { return Instance._scenarioEditManager; } }

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
