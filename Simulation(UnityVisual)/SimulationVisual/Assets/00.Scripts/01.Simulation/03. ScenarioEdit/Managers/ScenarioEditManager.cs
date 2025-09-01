using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class ScenarioEditManager
{
    public ScenarioEditInform scinfo;

    public GameObject ScenarioObject;
    public GameObject LoadedWaterObject;

    public string AgentState;

    public Dictionary<string, object> scenario_info;

    public Dictionary<string, object> scenario_terrianDict;
    public List<Dictionary<string, object>> scenario_agentListDict;
    public List<Dictionary<string, object>> scenario_initial_agentListDict;

    // 1. scenario button
    public Action CreateScenarioButtonAction;
    // 2. scenario classify(ScenarioSetting)
    public Action<Dictionary<string, object>> ScenarioInfoClassifyAction;
    // 3. scenario import(ScenarioImport.cs)
    public Action<string, string, string, Transform, Transform, string> ImportScenarioAction;
    // 3.1 scenario
    public Action<GameObject> ScenarioViewFitAction;
    public Action<string, string, string, Transform, string> ScenarioAgentButtonAction;
    public Action<string, string, string, Transform, Transform, string> ImportScenarioAgentSizeAction;
    public float maxFigure = 0f;
    public Action<string, string, string, Transform, Transform, string> ImportScenarioAgentAction;
    // 3.* agent add(in scenario)
    public Action<string, Dictionary<string, object>> AddAllocateButtonAction;
    // 4. scenario re save()
    public Action<string, string, string[], string[]> ScenarioReSaveAction;

    // Waypoint Action & Value
    public List<GameObject> waypoints = new List<GameObject>();
    public Action<GameObject, GameObject> WaypointConnectAction;
    public Action<GameObject, int> WaypointDisConnectionAction;
    public Action<GameObject, int, string> WaypointReConnectionAction;

    // Envrionment Action
    public float BuoyancyStrength = 1;
    public Action<string, string> UpdateWaterEnvrionmentAction;


    // Scenario Manage
    public string ScenarioId;
    // public string Scenario_GLBId;
    public List<string> onlyInInitialAgentIds = new List<string>();
    public List<string> onlyInCurrentAgentIds = new List<string>();

    public Action<string, Dictionary<string, object>> Scenario_GLBUpdateAction;
    
    public bool Editor_AGENT = false;

    // 1
    public void CreateScenarioButton()
    {
        CreateScenarioButtonAction?.Invoke();
    }

    // 2
    public void ClassifyScenarioInfo(Dictionary<string, object> ScenarioInfo)
    {
        ScenarioInfoClassifyAction?.Invoke(ScenarioInfo);
    }

    // 3 (2번 중간에 실행될 예정정)
    public void ImportScenario(string fileId, string fileName, string fileDesc, Transform Position, Transform Parent ,string table)
    {
        ImportScenarioAction?.Invoke(fileId, fileName, fileDesc, Position, Parent, table);
    }

    // 3.1 scenario glb 생성 및 맵 크기 맞추기
    public void ScenarioViewFit(GameObject SimulationEnvObject)
    {
        ScenarioViewFitAction?.Invoke(SimulationEnvObject);
    }
    // 3.1 scenario glb 의 agent 추출
    public void ScenarioAgentButton(string fileId, string fileName, string fileDesc, Transform Position, string table)
    {
        ScenarioAgentButtonAction?.Invoke(fileId, fileName, fileDesc, Position, table);
    }
    // 3.1 scenario button agent 생성
    public void ImportScenarioAgentSize(string fileId, string fileName, string fileDesc, Transform Position, Transform Parent ,string table)
    {
        ImportScenarioAgentSizeAction?.Invoke(fileId, fileName, fileDesc, Position, Parent, table);
    }

    public void ConnectWaypoint(List<GameObject> waypointsList)
    {
        GameObject way1 = waypointsList[0]; // 첫 번째 웨이포인트
        GameObject way2 = waypointsList[1]; // 두 번째 웨이포인트

        WaypointConnectAction?.Invoke(way1, way2);
    }

}
