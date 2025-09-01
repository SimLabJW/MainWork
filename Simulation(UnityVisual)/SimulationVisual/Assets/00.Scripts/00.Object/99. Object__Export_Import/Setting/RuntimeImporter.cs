using UnityEngine;
using System;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using GLTFast;
using System.Collections;
using System.Collections.Generic; // List, Dictionary를 위해 추가

public class RuntimeImporter : MonoBehaviour
{
    // 오브젝트와 정보 매핑용 딕셔너리
    public static Dictionary<GameObject, PrefabInfo.ImportedObjectInfo> importedObjectInfos = new Dictionary<GameObject, PrefabInfo.ImportedObjectInfo>();
    private GameObject gltfast_object = null;
    private void Start()
    {
        GameManager.createScenario.ImportEnvAction -= RuntimeImportFunction;
        GameManager.createScenario.ImportEnvAction += RuntimeImportFunction;

        GameManager.createScenario.ImportAgentAction -= RuntimeImportFunction;
        GameManager.createScenario.ImportAgentAction += RuntimeImportFunction;
    }

    public async void RuntimeImportFunction(string fileId, string fileName, string fileDesc, Transform Position, Transform Parent, string table)
    {
        Vector3 importPos = Position.position;

        await ImportModel(fileId, fileName, fileDesc, importPos, Parent, table);
    }

    public async Task ImportModel(string fileId, string fileName, string fileDesc, Vector3 Position, Transform Parent, string table)
    {
        var tcs = new TaskCompletionSource<bool>();
        string urlResult = null;

        IEnumerator WaitForComm()
        {
            GameManager.communication.Communication(table, new List<string> { "data" }, new Dictionary<string, object> { { "id", fileId }}, "GET");
            yield return new WaitForSeconds(1f);
            tcs.SetResult(true);
        }

        // 코루틴 시작
        StartCoroutine(WaitForComm());

        // 통신이 끝날 때까지 대기
        await tcs.Task;

        urlResult = GameManager.communication.Url_result;


        byte[] glbBytes = null;
        try
        {
            glbBytes = Convert.FromBase64String(urlResult);
        }
        catch (Exception e)
        {
            Debug.LogError($"GLB base64 디코딩 실패: {e.Message}");
            return;
        }

        var gltf = new GltfImport();
        bool success = false;
        try
        {
            success = await gltf.LoadGltfBinary(glbBytes);
        }
        catch (Exception e)
        {
            Debug.LogError($"GLB 로드 실패: {e.Message}");
            return;
        }

        if (!success)
        {
            Debug.LogError("GLB 로드에 실패했습니다.");
            return;
        }

        GameObject glbObject = new GameObject(fileName);
        glbObject.transform.SetParent(Parent, false);
        glbObject.transform.position = Position;

        GameManager.createScenario.currentObeject = glbObject;

        await gltf.InstantiateMainSceneAsync(glbObject.transform);

        // info 객체를 생성하여 딕셔너리에 저장
        PrefabInfo.ImportedObjectInfo info = new PrefabInfo.ImportedObjectInfo(fileId, fileName, "-",table);
        

        if(table == "Terrian")
        {
            WaterObjectSetting(info);
        }
        else if(table == "Agent")
        {
            AgentObjectSetting(info);
        }

    }

    IEnumerator FloaterToSeaAgent(GameObject AgentObject)
    {
        if (AgentObject.GetComponent<FloaterCreate>() == null)
        {
            AgentObject.AddComponent<FloaterCreate>();

            yield return new WaitForSeconds(0.2f);
            GameManager.createScenario.StartFloaterToSeaAgent(GameManager.createScenario.LoadedWaterObject, AgentObject);
        }
        
    }


    public void WaterObjectSetting(PrefabInfo.ImportedObjectInfo info)
    {
        // glbObject의 최상위 오브젝트만 추출
        Transform topObject = GameManager.createScenario.currentObeject.transform.childCount > 0 ? GameManager.createScenario.currentObeject.transform.GetChild(0) : null;

        if (topObject != null)
        {
            // water 오브젝트를 topObject의 위치, 회전, 부모로 생성
            GameObject water = GameObject.Instantiate(GameManager.Instance.Ocean, topObject.position, topObject.rotation, topObject.parent);

            // water의 이름을 topObject의 이름으로 변경
            water.name = topObject.name;

            // topObject의 모든 자식들을 water로 이동
            while (topObject.childCount > 0)
            {
                Transform child = topObject.GetChild(0);
                child.SetParent(water.transform, true);
            }

            GameManager.createScenario.LoadedWaterObject = water;

            PrefabInfo.AddImportedObjectInfo(GameManager.createScenario.LoadedWaterObject, info);

            GameObject.DestroyImmediate(topObject.gameObject);
        }
    }

    public void AgentObjectSetting(PrefabInfo.ImportedObjectInfo info)
    {
        // PrefabInfo의 딕셔너리에 저장
        PrefabInfo.AddImportedObjectInfo(GameManager.createScenario.currentObeject, info);

        GameManager.createScenario.currentObeject.AddComponent<Rigidbody>();

        MeshFilter[] meshFilters = GameManager.createScenario.currentObeject.GetComponentsInChildren<MeshFilter>();

        Matrix4x4 rootMatrix = GameManager.createScenario.currentObeject.transform.worldToLocalMatrix;
        CombineInstance[] combine = new CombineInstance[meshFilters.Length];

        for (int i = 0; i < meshFilters.Length; i++)
        {
            combine[i].mesh = meshFilters[i].sharedMesh;
            combine[i].transform = rootMatrix * meshFilters[i].transform.localToWorldMatrix;
        }

        Mesh combinedMesh = new Mesh();
        combinedMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32; // 큰 mesh 지원
        combinedMesh.CombineMeshes(combine);

        MeshCollider meshCollider = GameManager.createScenario.currentObeject.AddComponent<MeshCollider>();
        meshCollider.sharedMesh = combinedMesh;
        meshCollider.convex = true; // 필요에 따라 true로 변경

        StartCoroutine(FloaterToSeaAgent(GameManager.createScenario.currentObeject));
    }


}
