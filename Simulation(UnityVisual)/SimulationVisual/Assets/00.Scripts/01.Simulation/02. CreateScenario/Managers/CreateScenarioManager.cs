using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CreateScenarioManager
{
    public CreateScenarioInform csminfo;

    public GameObject currentObeject;
    public GameObject LoadedWaterObject;

    // RuntimeImporter Object Manage
    public bool LoadMap = false;
    public bool LoadAgent = false;
    public bool SaveObejct = false;
    
    // Load Object Event(by RuntimeImporter, env)
    public Action<string, string, Transform, Transform, string> ImportEnvAction;

    // Editor View Control Action(by EditorView)
    public Action<GameObject> EditorViewFitAction;
    // Phase 1 Object Event(EditorView, agent)
    public Action<string, string, Transform, string> EditorViewControlAction;
    // Phase 2 Object Size(by RuntimeImporter, agent_size)
    public Action<string, string, Transform, Transform, string> ImportAgentSizeAction;
    public float maxFigure = 0f;
    // Phase 3 Load Object(by RuntimeImporter, agent)
    public Action<string, string, Transform, Transform, string> ImportAgentAction;

    // Agent Function Action
    public Action<GameObject, GameObject> SeaAgentAction;
    // Scenario Save
    public Action<string, string, string[], string[]> ScenarioSaveAction;

    public string SaveScenarioName;
    public string SaveScenarioDescription;

    public bool Editor_ENV = false;
    public bool Editor_AGENT = false;
    // RuntimeImporter에서 RuntimeImportFunction 호출
    public void ImportObject(string fileId, string fileName, Transform Position, Transform Parent ,string table)
    {
        switch(table)
        {
            case "Terrian":
                ImportEnvAction?.Invoke(fileId, fileName, Position, Parent, table);
                break;
            case "Agent":
                ImportAgentSizeAction?.Invoke(fileId, fileName, Position, Parent, table);
                break;
        }
    }

    // Editor에서 Env 위치 및 Fit하고 화면 맞추기 EditorView 함수 활성화
    public void AddEditorViewFit(GameObject SimulationEnvObject)
    {
        // 화면 맞추기
        EditorViewFitAction?.Invoke(SimulationEnvObject);
    }
    public void EditorViewControl(string fileId, string fileName, Transform Position, string table)
    {
        // Agent 크기 및 위치 
        EditorViewControlAction?.Invoke(fileId, fileName, Position, table);
    }

    // RuntimeImporter.cs 통해 Sea_Agent 생성시 Floater 위치 지정
    public void StartFloaterToSeaAgent(GameObject WaterObject, GameObject AgentObject)
    {
        SeaAgentAction?.Invoke(WaterObject, AgentObject);
    }

    // Scenario Save
    public void SaveScenario(string scenarioName, string scenarioDescription, string[] terrianList, string[] agentList)
    {
        ScenarioSaveAction?.Invoke(scenarioName, scenarioDescription, terrianList, agentList);
    }
}
