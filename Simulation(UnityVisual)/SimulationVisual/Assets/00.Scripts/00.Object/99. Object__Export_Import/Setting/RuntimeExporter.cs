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
    // public GameObject targetObject;
    // public string savePath = "C:/Users/USER/Desktop/HDRP_Folder/Scenario/";

    private PrefabInfo prefabinfo = new PrefabInfo();

    private void Start()
    {
        GameManager.createScenario.ScenarioSaveAction -= RuntimeExportFunction;
        GameManager.createScenario.ScenarioSaveAction += RuntimeExportFunction;
    }
    public async void RuntimeExportFunction(string scenarioName, string scenarioDescription, string[] terrianList, string[] agentList)
    {
        GameObject targetObject = GameManager.createScenario.csminfo.createScenarioInfo.Simulation_ENV?.gameObject;
        if (targetObject == null)
        {
            Debug.LogWarning("대상 오브젝트가 존재하지 않습니다.");  
            return;
        }

        // ExportSettings 설정 (바이너리 포맷)
        var exportSettings = new ExportSettings
        {
            Format = GltfFormat.Binary // .glb로 저장 설정
        };

        var exporter = new GameObjectExport(exportSettings);

        exporter.AddScene(new GameObject[] { targetObject });

        // GLTFast 6.x에서는 SaveGLBToMemory가 없으므로, ExportGLB 메서드를 사용하여 메모리 스트림에 저장
        byte[] glbBytes = null;
        using (var memoryStream = new MemoryStream())
        {
            bool success = await exporter.SaveToStreamAndDispose(memoryStream);
            if (success)
            {
                glbBytes = memoryStream.ToArray();
            }
        }

        if (glbBytes != null && glbBytes.Length > 0)
        {
            Debug.Log("GLB 바이너리 데이터 추출 성공, 크기: " + glbBytes.Length + " bytes");

            // 또는 glbBytes를 네트워크 전송, DB 저장 등 원하는 방식으로 활용
        }
        else
        {
            Debug.LogError("GLB 바이너리 데이터 추출에 실패했습니다.");
        }

        var tcs = new TaskCompletionSource<bool>();

        IEnumerator WaitForComm()
        {
            // terrianList와 agentList는 string[]이므로, 하나의 string으로 변환해서 전달해야 함
            string terrianListStr = string.Join(",", terrianList);
            string agentListStr = string.Join(",", agentList);

            // PrefabInfo.cs의 ScenarioDefaultEnvInfo 데이터도 같이 전달
            // (예시: JSON 문자열로 변환해서 전달)
            string scenarioDefaultEnvInfoJson = JsonUtility.ToJson(prefabinfo.scenarioDefaultEnvInfo);

            GameManager.communication.SaveCommunication(
                "Scenario",
                new string[] { scenarioName, scenarioDescription, terrianListStr, agentListStr, Convert.ToBase64String(glbBytes), scenarioDefaultEnvInfoJson }
            );
            yield return new WaitForSeconds(1f);
            tcs.SetResult(true);
        }

        // 코루틴 시작
        StartCoroutine(WaitForComm());

        // 통신이 끝날 때까지 대기
        await tcs.Task;
    }

    
}





