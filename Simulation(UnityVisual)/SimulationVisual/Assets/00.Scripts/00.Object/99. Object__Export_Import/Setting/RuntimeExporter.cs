using UnityEngine;
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using GLTFast.Export;
using Newtonsoft.Json;
using System.Threading.Tasks;

public class RuntimeExporter : MonoBehaviour
{
    private void Start()
    {
        GameManager.createScenario.ScenarioSaveAction -= RuntimeExportFunction;
        GameManager.createScenario.ScenarioSaveAction += RuntimeExportFunction;
    }
    public async void RuntimeExportFunction(string scenarioName, string scenarioDescription, string[] terrianList, string[] agentList) 
    { 
        // Simulation_ENV의 이름을 scenarioName으로 임시 변경
        GameObject simulationEnv = GameManager.createScenario.csminfo.createScenarioInfo.Simulation_ENV?.gameObject;
        string originalName = null;
        if (simulationEnv != null)
        {
            originalName = simulationEnv.name;
            simulationEnv.name = scenarioName;
        }

        var exportSettings = new ExportSettings 
        { 
            Format = GltfFormat.Binary // .glb로 저장 설정 
        }; 
        
        var exporter = new GameObjectExport(exportSettings); 
        exporter.AddScene(new GameObject[] { simulationEnv }); 

        byte[] glbBytes = null; 
        using (var memoryStream = new MemoryStream()) 
        { 
            bool success = await exporter.SaveToStreamAndDispose(memoryStream); 
            if (success) { glbBytes = memoryStream.ToArray(); } 
        } 
        if (glbBytes != null && glbBytes.Length > 0) 
        { 
            Debug.Log("GLB 바이너리 데이터 추출 성공, 크기: " + glbBytes.Length + " bytes"); // 또는 glbBytes를 네트워크 전송, DB 저장 등 원하는 방식으로 활용 
        } 
        else { Debug.LogError("GLB 바이너리 데이터 추출에 실패했습니다."); } 
        
        var tcs = new TaskCompletionSource<bool>(); 
        
        IEnumerator WaitForComm() 
        { 
            string terrianListStr = string.Join(",", terrianList); 
            string agentListStr = string.Join(",", agentList); 
            GameManager.communication.SaveCommunication( "Scenario", new string[] { scenarioName, scenarioDescription, terrianListStr, agentListStr, Convert.ToBase64String(glbBytes)} ); 
            yield return new WaitForSeconds(1f); 
            tcs.SetResult(true); 
        } // 코루틴 시작 
            
        StartCoroutine(WaitForComm()); // 통신이 끝날 때까지 대기 
        await tcs.Task;

        // 이름을 원래대로 복구
        if (simulationEnv != null && originalName != null)
        {
            simulationEnv.name = originalName;
        }

        if (simulationEnv != null) 
        { 
            for (int i = simulationEnv.transform.childCount - 1; i >= 0; i--)
            { 
                GameObject child = simulationEnv.transform.GetChild(i).gameObject; 
                Destroy(child); 
            } 
        }
    }

    
}





