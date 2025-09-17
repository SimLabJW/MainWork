using UnityEngine;
using System;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using GLTFast;
using System.Collections;
using System.Collections.Generic; // List, Dictionary를 위해 추가

public class ScenarioImporter : MonoBehaviour
{
    // 오브젝트와 정보 매핑용 딕셔너리
    public static Dictionary<GameObject, PrefabInfo.ImportedObjectInfo> importedObjectInfos = new Dictionary<GameObject, PrefabInfo.ImportedObjectInfo>();
    private GameObject gltfast_object = null;
    private PrefabInfo.ImportedObjectInfo info;
    private PrefabInfo.ImportObjectUnityInfo unity_info;

    private void Start()
    {
        GameManager.scenarioEdit.ImportScenarioAction -= RuntimeImportFunction;
        GameManager.scenarioEdit.ImportScenarioAction += RuntimeImportFunction;

        GameManager.scenarioEdit.ImportScenarioAgentAction -= RuntimeImportFunction;
        GameManager.scenarioEdit.ImportScenarioAgentAction += RuntimeImportFunction;
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


        await gltf.InstantiateMainSceneAsync(glbObject.transform);

        if(table == "Agent" && GameManager.scenarioEdit.ScenarioObject is not null)
        {
            // info 객체를 생성하여 딕셔너리에 저장
            info = new PrefabInfo.ImportedObjectInfo(fileId, fileName, fileDesc, table);
            StartCoroutine(AgentObjectSetting(info, glbObject));   

            // fileDesc가 null이거나 빈 문자열이면 "설명 없음"으로 출력
            string agentDesc = string.IsNullOrEmpty(fileDesc) ? "설명 없음" : fileDesc;
            // Debug.Log($"agent_id: {fileId}, agent_name: {fileName}, agent_desc: {agentDesc}");
            var agentInfo = new Dictionary<string, object>
            {
                { "agent_id", fileId },
                { "agent_name", fileName },
                { "agent_desc", agentDesc }
            };
            GameManager.scenarioEdit.scenario_agentListDict.Add(agentInfo);
            
        }
        else if(table == "Scenario")
        {
            // 1. gltf가 생성한 자식들만 따로 저장
            List<Transform> loadedChildren = new List<Transform>();
            foreach (Transform child in glbObject.transform)
            {
                loadedChildren.Add(child);
            }

            // 2. 자식들을 glbObject로 이동
            foreach (Transform child in loadedChildren)
            {
                child.SetParent(glbObject.transform.parent, true); 
            }

            // 3. 중복된 오브젝트 삭제
            GameObject.DestroyImmediate(glbObject);

            // 4. currentObject를 새로 지정
            GameManager.scenarioEdit.ScenarioObject = Parent.gameObject.transform.GetChild(0).gameObject;

            GameObject Scenario_Terrian = null;

            foreach(var agent in GameManager.scenarioEdit.scenario_agentListDict)
            {
                foreach (Transform scenario_object in GameManager.scenarioEdit.ScenarioObject.transform)
                {
                    if (agent.ContainsKey("agent_name") && agent["agent_name"]?.ToString() == scenario_object.name)
                    {
                        // agent_id와 agent_name을 추출하여 info 객체 생성
                        string agentId = agent.ContainsKey("agent_id") ? agent["agent_id"]?.ToString() : "";
                        string agentName = agent.ContainsKey("agent_name") ? agent["agent_name"]?.ToString() : "";
                        string agentDesc = agent.ContainsKey("agent_desc") ? agent["agent_desc"]?.ToString() : "";

                        info = new PrefabInfo.ImportedObjectInfo(agentId, agentName, agentDesc, "Agent");
                        
                        StartCoroutine(AgentObjectSetting(info, scenario_object.gameObject));

                        
                    }
                }
            }
            foreach (Transform scenario_object in GameManager.scenarioEdit.ScenarioObject.transform)
            {
                foreach (var terrianPair in GameManager.scenarioEdit.scenario_terrianDict)
                {
                    // KeyValuePair의 Key와 Value 속성을 사용
                    if (terrianPair.Key == "terrian_name" && terrianPair.Value?.ToString() == scenario_object.name)
                    {
                        string terrianId = GameManager.scenarioEdit.scenario_terrianDict.ContainsKey("terrian_id") ? GameManager.scenarioEdit.scenario_terrianDict["terrian_id"]?.ToString() : "";
                        string terrianName = GameManager.scenarioEdit.scenario_terrianDict.ContainsKey("terrian_name") ? GameManager.scenarioEdit.scenario_terrianDict["terrian_name"]?.ToString() : "";
                        string terrianDesc = GameManager.scenarioEdit.scenario_terrianDict.ContainsKey("terrian_desc") ? GameManager.scenarioEdit.scenario_terrianDict["terrian_desc"]?.ToString() : "";

                        info = new PrefabInfo.ImportedObjectInfo(terrianId, terrianName, terrianDesc,"Terrian");
                        Scenario_Terrian = scenario_object.gameObject;

        
                        WaterObjectSetting(info, scenario_object.gameObject);
                    }
                }
            }

        }

    }

    public void WaterObjectSetting(PrefabInfo.ImportedObjectInfo info , GameObject Terrian_Object)
    {
        // glbObject의 최상위 오브젝트만 추출
        Transform TerrianObject = Terrian_Object.transform.childCount > 0 ? Terrian_Object.transform : null;

        if (TerrianObject != null)
        {
            // water 오브젝트를 topObject의 위치, 회전, 부모로 생성
            GameObject water = GameObject.Instantiate(GameManager.Instance.Ocean, TerrianObject.position, TerrianObject.rotation, TerrianObject.parent);

            // water의 이름을 topObject의 이름으로 변경
            water.name = TerrianObject.name;

            // topObject의 모든 자식들을 water로 이동
            while (TerrianObject.childCount > 0)
            {
                Transform child = TerrianObject.GetChild(0);
                child.SetParent(water.transform, true);
            }

            GameManager.scenarioEdit.LoadedWaterObject = water;

            PrefabInfo.AddImportedObjectInfo(GameManager.scenarioEdit.LoadedWaterObject, info);

            GameObject.DestroyImmediate(TerrianObject.gameObject);
        }
    }

    IEnumerator AgentObjectSetting(PrefabInfo.ImportedObjectInfo info, GameObject AgentObject)
    {
        // PrefabInfo의 딕셔너리에 저장(기본값 - db기반)
        PrefabInfo.AddImportedObjectInfo(AgentObject, info);
        // PrefabInfo의 딕셔너리에 저장(기본값 - unity 기반)
        string unityId = AgentObject.GetInstanceID().ToString();
        unity_info = new PrefabInfo.ImportObjectUnityInfo(unityId, GameManager.scenarioEdit.AgentState);
        PrefabInfo.AddImportedObjectUnityInfo(AgentObject, unity_info);

        var agentInfo = new Dictionary<string, object>
        {
            { "agent_id", info.fileId },
            { "agent_name", info.fileName },
            { "agent_desc", info.fileDesc },
            { "unity_id", unityId }
        };
        GameManager.scenarioEdit.AddAllocateButtonAction?.Invoke("agent", agentInfo);

        // Rigidbody가 이미 존재하는지 확인 후 없을 때만 추가
        if (AgentObject.GetComponent<Rigidbody>() == null)
        {
            AgentObject.AddComponent<Rigidbody>();
        }

        MeshFilter[] meshFilters = AgentObject.GetComponentsInChildren<MeshFilter>();

        Matrix4x4 rootMatrix = AgentObject.transform.worldToLocalMatrix;
        CombineInstance[] combine = new CombineInstance[meshFilters.Length];

        for (int i = 0; i < meshFilters.Length; i++)
        {
            combine[i].mesh = meshFilters[i].sharedMesh;
            combine[i].transform = rootMatrix * meshFilters[i].transform.localToWorldMatrix;
        }

        Mesh combinedMesh = new Mesh();
        combinedMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32; // 큰 mesh 지원
        combinedMesh.CombineMeshes(combine);

        MeshCollider meshCollider = AgentObject.GetComponent<MeshCollider>();
        if (meshCollider == null)
        {
            meshCollider = AgentObject.AddComponent<MeshCollider>();
        }
        meshCollider.sharedMesh = combinedMesh;
        meshCollider.convex = true; // 필요에 따라 true로 변경

        // 에이전트 정면 방향의 가장 높은 위치에 카메라 설치/업데이트
        CreateOrUpdateAgentTopFrontCamera(AgentObject);

        yield return new WaitForSeconds(0.1f);

        StartCoroutine(FloaterToSeaAgent(AgentObject));
    }

    IEnumerator FloaterToSeaAgent(GameObject AgentObject)
    {
        if (AgentObject.GetComponent<FloaterCreate>() == null)
        {
            AgentObject.AddComponent<FloaterCreate>();

            yield return new WaitForSeconds(0.2f);
            GameManager.createScenario.StartFloaterToSeaAgent(GameManager.scenarioEdit.LoadedWaterObject, AgentObject);
        }
        
    }

    // 에이전트의 메시 바운드를 기준으로 정면(Forward) 방향의 가장 높은 지점에 카메라 배치
    void CreateOrUpdateAgentTopFrontCamera(GameObject agent)
    {
        if (agent == null) return;

        // 이미 존재하면 재배치만 수행
        Transform existing = agent.transform.Find("AgentTopFrontCamera");

        // 모든 Renderer 바운드를 합쳐 전체 바운드 계산
        Renderer[] renderers = agent.GetComponentsInChildren<Renderer>();
        if (renderers == null || renderers.Length == 0) return;

        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }

        Vector3 center = bounds.center;
        // 오브젝트 중심에서 가장 높은 점(Y max)에 위치
        Vector3 cameraPos = new Vector3(center.x, bounds.max.y, center.z);

        if (existing == null)
        {
            GameObject camObj = new GameObject("AgentTopFrontCamera");
            camObj.transform.SetParent(agent.transform, worldPositionStays: true);
            camObj.transform.position = cameraPos;
            camObj.transform.rotation = Quaternion.LookRotation(agent.transform.forward, Vector3.up);

            var cam = camObj.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.Skybox;
            cam.nearClipPlane = 0.1f;
            cam.farClipPlane = 10000f;
            cam.fieldOfView = 60f;
        }
        else
        {
            existing.position = cameraPos;
            existing.rotation = Quaternion.LookRotation(agent.transform.forward, Vector3.up);
        }
    }

}
