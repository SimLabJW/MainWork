using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimulationManager
{
    public SimulationInform sm;

    public GameObject currentObeject;
    public GameObject LoadedWaterObject;
    //public GameObject LoadedAgentObject1;
    //public GameObject LoadedAgentObject2;

    public GameObject[] PointArea;

    // RuntimeImporter Object Manage
    public bool LoadMap = false;
    public bool LoadAgent = false;
    public bool SaveObejct = false;
    
    // Load Object Event(by RuntimeImporter, env)
    public Action<string, string, Transform, Transform> ImportEnvAction;

    // Editor View Control Action(by EditorView)
    public Action<GameObject> EditorViewFitAction;
    // Phase 1 Object Event(EditorView, agent)
    public Action<string, string, Transform> EditorViewControlAction;
    // Phase 2 Object Size(by RuntimeImporter, agent_size)
    public Action<string, string, Transform, Transform> ImportAgentSizeAction;
    public float maxFigure = 0f;
    // Phase 3 Load Object(by RuntimeImporter, agent)
    public Action<string, string, Transform, Transform> ImportAgentAction;

    // Phase 4 Create Scenario
    public Action CreatePointArea;

    // Agent Function Action
    public Action<GameObject, GameObject> SeaAgentAction;

    // Editor Object deactivate -> Visul Object activate
    public Action NextPhase;

    public bool Editor_ENV = false;
    public bool Editor_AGENT = false;
    // RuntimeImporter의 RuntimeImportFunction 실행
    public void ImportObject(string path, string fileName, Transform Position, Transform Parent ,string mode)
    {
        switch(mode)
        {
            case "env":
                ImportEnvAction?.Invoke(path, fileName, Position, Parent);
                break;
            case "agent_size":
                ImportAgentSizeAction?.Invoke(path, fileName, Position, Parent);
                break;
            case "agent_import":
                Editor_AGENT = true;
                ImportAgentAction?.Invoke(path, fileName, Position, Parent);
                break;
        }
    }

    // Editor에서 Env 설치 후 Fit하게 화면 맞추기 EditorView 기능 활성화
    public void AddEditorViewFit(GameObject SimulationEnvObject)
    {
        // 화면 정렬
        EditorViewFitAction?.Invoke(SimulationEnvObject);
    }
    public void EditorViewControl(string path, string fileName, Transform Position)
    {
        // Agent 크기 및 배치 
        EditorViewControlAction?.Invoke(path, fileName, Position);
    }

    // RuntimeImporter.cs 에서 Sea_Agent 생성시 Floater 설치 실행
    public void StartFloaterToSeaAgent(GameObject WaterObject, GameObject AgentObject)
    {
        SeaAgentAction?.Invoke(WaterObject, AgentObject);
    }
}
