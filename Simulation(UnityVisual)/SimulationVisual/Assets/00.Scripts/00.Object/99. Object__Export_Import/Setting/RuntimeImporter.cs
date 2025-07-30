using UnityEngine;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using GLTFast;
using System.Collections;

public class RuntimeImporter : MonoBehaviour
{
    private PrefabInfo prefabinfo = new PrefabInfo();
    private ObjectProfile profile = new ObjectProfile();

    private GameObject gltfast_object = null;
    private void Start()
    {
        GameManager.simulation.ImportEnvAction -= RuntimeImportFunction;
        GameManager.simulation.ImportEnvAction += RuntimeImportFunction;

        GameManager.simulation.ImportAgentAction -= RuntimeImportFunction;
        GameManager.simulation.ImportAgentAction += RuntimeImportFunction;

        
    }
    public async void RuntimeImportFunction(string path, string fileName, Transform Position, Transform Parent)
    {
        string glbPath = Path.Combine(path, fileName + ".glb");
        string jsonPath = Path.Combine(path, fileName + ".json");

        Vector3 importPos = Position.position;

        await ImportModel(jsonPath, glbPath, importPos, Parent);
    }
    public async Task ImportModel(string jsonPath, string glbPath, Vector3 Position, Transform Parent)
    {
        var gltf = new GltfImport();

        // 1. ���� �ڽ� ���� ���
        int childCountBefore = Parent.childCount;

        bool success = await gltf.Load(glbPath);
        if (!success)
        {
            Debug.LogError("GLB �ҷ����� ����: " + glbPath);
            return;
        }
        success = await gltf.InstantiateMainSceneAsync(Parent);
        if (!success)
        {
            Debug.LogError("�ν��Ͻ�ȭ ����");
            return;
        }

        Debug.Log("GLB �ҷ����� ����: " + glbPath);

        // 2. ���� �߰��� �ڽĸ� Ž��
        for (int i = childCountBefore; i < Parent.childCount; i++)
        {
            Transform child = Parent.GetChild(i);
            string containName = child.name;

            if (containName.Contains("Map"))
            {
                GameManager.simulation.LoadMap = true;
                gltfast_object = child.gameObject;

                break;
            }
            else if (containName.Contains("Agent"))
            {
                gltfast_object = child.gameObject;
                break;
            }
            
        }
        GameManager.simulation.currentObeject = gltfast_object;

        // 3. json ���� �� ��ġ ����
        if (GameManager.simulation.currentObeject != null)
        {
            ApplyPrefabInfoFromJson(jsonPath, GameManager.simulation.currentObeject);
            GameManager.simulation.currentObeject.transform.position = Position;

            // 4. Profile Activate
            Object_Profile(GameManager.simulation.currentObeject);

            // 5. ��ɺз�
            Classify_Map_Object(GameManager.simulation.currentObeject);
            Classify_Agent_Object(GameManager.simulation.currentObeject);
        }


        //GameManager.simulation.currentObeject = null;
    }

    public void Object_Profile(GameObject go)
    {
        if (go.tag == "Map")
        {
            GameManager.simulation.sm.simulationInfo.editorInform.env_profile.Env_Profile.SetActive(true);
            GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.Agent_Profile.SetActive(false);
            GameManager.simulation.sm.simulationInfo.editorInform.scenario_profile.Scenario_Profile.SetActive(false);

            profile.MapProfileInputFieldApply(go);
        }
        else if (go.tag.Contains("Agent"))
        {
            GameManager.simulation.sm.simulationInfo.editorInform.env_profile.Env_Profile.SetActive(false);
            GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.Agent_Profile.SetActive(true);
            GameManager.simulation.sm.simulationInfo.editorInform.scenario_profile.Scenario_Profile.SetActive(false);

            profile.AgentProfileInputFielApply(go);
        }
    }

    public void Classify_Map_Object(GameObject go)
    {
        if (go.tag == "Map")
        {
            // Map �Ʒ� ��� ���� ������Ʈ�� Ž��
            Transform[] allChildren = go.GetComponentsInChildren<Transform>();
            foreach (Transform t in allChildren)
            {
                if (t.CompareTag("Water"))
                {
                    GameManager.simulation.LoadedWaterObject = t.gameObject;
                    break;
                }
            }
        }
    }
    public void Classify_Agent_Object(GameObject go)
    {
        if (go.tag == "Sea_Agent")
        {
            StartCoroutine(FloaterToSeaAgent(go));
        }
    }
    IEnumerator FloaterToSeaAgent(GameObject AgentObject)
    {
        if (AgentObject.GetComponent<FloaterCreate>() == null)
        {
            AgentObject.AddComponent<FloaterCreate>();

            yield return new WaitForSeconds(0.2f);
            GameManager.simulation.StartFloaterToSeaAgent(GameManager.simulation.LoadedWaterObject, AgentObject);
        }
        
    }

    private void ApplyPrefabInfoFromJson(string jsonPath, GameObject gltfast_object)
    {

        if (!File.Exists(jsonPath))
        {
            Debug.LogWarning("JSON ���� ����: " + jsonPath);
            return;
        }

        string json = File.ReadAllText(jsonPath);
        var objectInfo = JsonConvert.DeserializeObject<PrefabInfo.ObjectInfo>(json);

        prefabinfo.ApplyInfo(objectInfo, gltfast_object);
        Debug.Log("JSON ���� ���� �Ϸ�");
    }

}
