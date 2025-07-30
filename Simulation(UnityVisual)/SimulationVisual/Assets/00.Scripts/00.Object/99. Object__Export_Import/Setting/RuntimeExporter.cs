using UnityEngine;
using System.IO;
using GLTFast.Export;
using Newtonsoft.Json;

public class RuntimeExporter : MonoBehaviour
{
    public GameObject targetObject;
    public string savePath = "C:/Users/USER/Desktop/HDRP_Folder/Create/Construction/";

    private PrefabInfo prefabinfo = new PrefabInfo();

    async void Start()
    {
        if (targetObject == null)
        {
            Debug.LogWarning("내보낼 대상이 없습니다.");
            return;
        }

        if (!Directory.Exists(savePath))
        {
            Directory.CreateDirectory(savePath);
        }

        // object json file
        var object_info = prefabinfo.BuildInfo(targetObject);
        string json = JsonConvert.SerializeObject(
            object_info,
            Formatting.Indented,
            new JsonSerializerSettings
            {
                ReferenceLoopHandling = ReferenceLoopHandling.Ignore
            }
        );
        File.WriteAllText(Path.Combine(savePath, targetObject.name + ".json"), json);

        // object glb file
        string fullPath = Path.Combine(savePath, targetObject.name + ".glb");

        var exportSettings = new ExportSettings
        {
            Format = GltfFormat.Binary // .glb로 단일 저장
        };

        var exporter = new GameObjectExport(exportSettings);

        exporter.AddScene(new GameObject[] { targetObject });

        bool success = await exporter.SaveToFileAndDispose(fullPath); 


        if (success)
            Debug.Log("GLB 내보내기 성공: " + fullPath);
        else
            Debug.LogError("내보내기 실패");
    }

    
}





