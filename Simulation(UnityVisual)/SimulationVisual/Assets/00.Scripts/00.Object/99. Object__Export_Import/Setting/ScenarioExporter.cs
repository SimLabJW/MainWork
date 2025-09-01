using UnityEngine;
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using GLTFast.Export;
using Newtonsoft.Json;
using System.Threading.Tasks;

public class ScenarioExporter : MonoBehaviour
{
    private void Start()
    {
        GameManager.scenarioEdit.Scenario_GLBUpdateAction -= UpdateExportFunction;
        GameManager.scenarioEdit.Scenario_GLBUpdateAction += UpdateExportFunction;
    }
    public async void UpdateExportFunction(string table, Dictionary<string, object> filters) 
    { 

        var exportSettings = new ExportSettings 
        { 
            Format = GltfFormat.Binary // .glb로 저장 설정 
        }; 
        
        var exporter = new GameObjectExport(exportSettings); 
        exporter.AddScene(new GameObject[] { GameManager.scenarioEdit.ScenarioObject }); 

        byte[] glbBytes = null; 
        using (var memoryStream = new MemoryStream()) 
        { 
            bool success = await exporter.SaveToStreamAndDispose(memoryStream); 
            if (success) { glbBytes = memoryStream.ToArray(); } 
        } 

        
        var tcs = new TaskCompletionSource<bool>(); 
        
        IEnumerator WaitForComm() 
        { 
            GameManager.communication.ScenarioUpdate("GLB", new Dictionary<string, object> { { "data", Convert.ToBase64String(glbBytes) } }, filters);
            yield return new WaitForSeconds(1f); 
            tcs.SetResult(true); 
        } // 코루틴 시작 
            
        StartCoroutine(WaitForComm()); // 통신이 끝날 때까지 대기 
        await tcs.Task;

    }
}
