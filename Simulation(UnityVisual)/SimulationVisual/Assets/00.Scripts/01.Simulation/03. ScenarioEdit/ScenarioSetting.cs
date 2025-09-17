using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI; // UnityEngine.UI 추가

public class ScenarioSetting : MonoBehaviour
{
    
    // Start is called before the first frame update
    void Start()
    {
        GameManager.scenarioEdit.ScenarioInfoClassifyAction -= ScenarioClassify;
        GameManager.scenarioEdit.ScenarioInfoClassifyAction += ScenarioClassify;

        GameManager.scenarioEdit.AddAllocateButtonAction -= AllocateFiletoButton;
        GameManager.scenarioEdit.AddAllocateButtonAction += AllocateFiletoButton;
    }

    public void ScenarioClassify(Dictionary<string, object> scenarioInfo)
    {
        // 시나리오 정보 분류용 딕셔너리 생성
        Dictionary<string, object> scenarioDict = new Dictionary<string, object>();
        Dictionary<string, object> terrianDict = new Dictionary<string, object>();
        Dictionary<string, object> envDict = new Dictionary<string, object>();
        List<Dictionary<string, object>> agentList = new List<Dictionary<string, object>>();

        // 시나리오 관련 키 분류
        string[] scenarioKeys = { "scenario_id", "scenario_name", "scenario_desc" };
        string[] terrianKeys = { "terrian_id", "terrian_name", "terrian_desc", "terrian_glb_id" };
        string[] envKeys = { "env_id", "env_name", "weather_type", "lighting_intensity", "sun_angle", "temperature", "rain_intensity", "visibility", "wave_height", "wave_speed", "wave_direction_ns", "wave_direction_we", "wave_clarity", "buoyancy_strength", "sea_level" };

        // 시나리오 정보 분류
        foreach (var key in scenarioKeys)
        {
            if (scenarioInfo.ContainsKey(key))
                scenarioDict[key] = scenarioInfo[key];
        }

        foreach (var key in terrianKeys)
        {
            if (scenarioInfo.ContainsKey(key))
                terrianDict[key] = scenarioInfo[key];
        }

        foreach (var key in envKeys)
        {
            if (scenarioInfo.ContainsKey(key))
                envDict[key] = scenarioInfo[key];
        }

        // agent 정보 분류
        if (scenarioInfo.ContainsKey("agents") && scenarioInfo["agents"] is List<Dictionary<string, object>>)
        {
            agentList = scenarioInfo["agents"] as List<Dictionary<string, object>>;
        }
        else if (scenarioInfo.ContainsKey("agents") && scenarioInfo["agents"] is IEnumerable<object>)
        {
            // 혹시 object 리스트로 들어온 경우 변환
            foreach (var agentObj in (IEnumerable<object>)scenarioInfo["agents"])
            {
                if (agentObj is Dictionary<string, object> agentDict)
                {
                    agentList.Add(agentDict);
                }
                else if (agentObj is Newtonsoft.Json.Linq.JObject jAgent)
                {
                    agentList.Add(jAgent.ToObject<Dictionary<string, object>>());
                }
            }
        }

        // 초기 에이전트 리스트는 깊은 복사로 저장 (원본 보존)
        GameManager.scenarioEdit.scenario_initial_agentListDict = new List<Dictionary<string, object>>();
        foreach (var agent in agentList)
        {
            var copiedAgent = new Dictionary<string, object>();
            foreach (var kvp in agent)
            {
                copiedAgent[kvp.Key] = kvp.Value;
            }
            GameManager.scenarioEdit.scenario_initial_agentListDict.Add(copiedAgent);
        }
        
        // 현재 에이전트 리스트는 원본 참조
        GameManager.scenarioEdit.scenario_agentListDict = agentList;
        
        GameManager.scenarioEdit.scenario_terrianDict = terrianDict;
        
        UpdateScenarioUI_Scenario(scenarioDict);
    
        StartCoroutine(DelayedFit());

        UpdateScenarioUI_Allocate(terrianDict, agentList);
        
        UpdateScenarioUI_DeplyablePlatforms();

        UpdateScenarioUI_Environment(envDict);
        
    }

    IEnumerator DelayedFit()
    {
        GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.gameObject.SetActive(true);
        yield return new WaitForSeconds(1.58f);
        GameManager.scenarioEdit.ScenarioViewFit(GameManager.scenarioEdit.LoadedWaterObject);
    }

    public void UpdateScenarioUI_Scenario(Dictionary<string, object> scenario)
    {
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.scenarioSet.ScenarioName.text = scenario["scenario_name"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.scenarioSet.ScenarioDescription.text = scenario["scenario_desc"]?.ToString();
        ImportScenarioToCreate(
            scenario["scenario_id"]?.ToString(),
            scenario["scenario_name"]?.ToString(),
            scenario["scenario_desc"]?.ToString(),
            GameManager.scenarioEdit.scinfo.scenarioEditInfo.Simulation_ENV,
            GameManager.scenarioEdit.scinfo.scenarioEditInfo.Simulation_ENV,
            "Scenario"
        );
    }


    public void UpdateScenarioUI_Environment(Dictionary<string, object> env)
    {

        if (env.ContainsKey("weather_type") && env["weather_type"] != null)
        {
            string weatherTypeStr = env["weather_type"].ToString();
            // Dropdown 컴포넌트 가져오기
            var dropdown = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WeatherType.GetComponent<Dropdown>();
            // 옵션 리스트에서 해당 값의 인덱스를 찾음
            int idx = dropdown.options.FindIndex(opt => opt.text == weatherTypeStr);
            if (idx >= 0)
            {
                dropdown.value = idx;
            }
            else
            {
                // 만약 값이 없으면 첫 번째 옵션(기본값)으로 설정
                dropdown.value = 0;
                Debug.LogWarning($"[환경설정] '{weatherTypeStr}' 값이 드롭다운에 없어 첫 번째 옵션(기본값)으로 설정합니다.");
            }
            dropdown.RefreshShownValue();

            // 실제로 값이 바뀌었는지 확인
            string nowText = dropdown.options[dropdown.value].text;
        }

        //18000
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.LightingIntensity_T.text = env["lighting_intensity"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.LightingIntensity_S.value = float.Parse(env["lighting_intensity"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("lighting_intensity", env["lighting_intensity"]?.ToString());

        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.SunAngle_T.text = env["sun_angle"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.SunAngle_S.value = float.Parse(env["sun_angle"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("sun_angle", env["sun_angle"]?.ToString());

        //6500
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.Temperature_T.text = env["temperature"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.Temperature_S.value = float.Parse(env["temperature"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("temperature", env["temperature"]?.ToString());

        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.RainIntensity_T.text = env["rain_intensity"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.RainIntensity_S.value = float.Parse(env["rain_intensity"].ToString());
        // GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_height", env["wave_height"]?.ToString());

        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.Visibility_T.text = env["visibility"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.Visibility_S.value = float.Parse(env["visibility"].ToString());
        // GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_height", env["wave_height"]?.ToString());

        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveHeight_T.text = env["wave_height"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveHeight_S.value = float.Parse(env["wave_height"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_height", env["wave_height"]?.ToString());

        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveSpeed_T.text = env["wave_speed"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveSpeed_S.value = float.Parse(env["wave_speed"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_speed", env["wave_speed"]?.ToString());

        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveDirectionNS.value = float.Parse(env["wave_direction_ns"].ToString());
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveDirectionWE.value = float.Parse(env["wave_direction_we"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_direction_ns", env["wave_direction_ns"]?.ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_direction_we", env["wave_direction_we"]?.ToString());

        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveClarity_T.text = env["wave_clarity"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveClarity_S.value = float.Parse(env["wave_clarity"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_clarity", env["wave_clarity"]?.ToString());

        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.BuoyancyStrength_T.text = env["buoyancy_strength"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.BuoyancyStrength_S.value = float.Parse(env["buoyancy_strength"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("buoyancy_strength", env["buoyancy_strength"]?.ToString());

        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.SeaLevel_T.text = env["sea_level"]?.ToString();
        GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.SeaLevel_S.value = float.Parse(env["sea_level"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("sea_level", env["sea_level"]?.ToString());

    }
    public void UpdateScenarioUI_Allocate(Dictionary<string, object> terrian, List<Dictionary<string, object>> agentList)
    {
        foreach (Transform child in GameManager.scenarioEdit.scinfo.scenarioEditInfo.AllocateObjectContent.transform)
        {
            Destroy(child.gameObject);
        }

        AllocateFiletoButton("terrian", terrian);
        // foreach (var agent in agentList)
        // {
        //     AllocateFiletoButton("agent",agent);
        // }
        
        // 모든 버튼 생성 후 규칙에 따라 delete 버튼 상태 동기화
        SyncAllocateDeleteButtons();
    }

    void AllocateFiletoButton(string table, Dictionary<string, object> FileList)
    {
        if (FileList == null || FileList.Count == 0)
        {
            return;
        }

        GameObject newButtonObj = Instantiate(
            GameManager.scenarioEdit.scinfo.scenarioEditInfo.AllocateObjectButton,
            GameManager.scenarioEdit.scinfo.scenarioEditInfo.AllocateObjectContent.transform
        );
        Button newButton = newButtonObj.GetComponent<Button>();
        // Button 컴포넌트를 특성(타입)으로 찾아서 가져오는 방법 예시
        
        Text nameText = newButtonObj.transform.Find("Name_T").GetComponent<Text>();
        Text descriptionText = newButtonObj.transform.Find("Description_T").GetComponent<Text>();
        Button stateImage = newButtonObj.transform.Find("Image").GetComponent<Button>();

        string fileId = FileList.ContainsKey($"{table}_id") ? FileList[$"{table}_id"]?.ToString() : "";
        string fileName = FileList.ContainsKey($"{table}_name") ? FileList[$"{table}_name"]?.ToString() : "";
        string fileDesc = FileList.ContainsKey($"{table}_desc") ? FileList[$"{table}_desc"]?.ToString() : "";

        nameText.text = fileName;
        descriptionText.text = fileDesc;

        Button DeleteButton = newButtonObj.transform.Find("Del").GetComponent<Button>();
        if (table == "terrian")
        {
            DeleteButton.interactable = false;
            var terrianImg = stateImage != null ? stateImage.GetComponent<Image>() : null;
            if (terrianImg != null) terrianImg.color = Color.gray;
        }
        else
        {
            // 초기 상태 및 색상 설정
            string currentState = GameManager.scenarioEdit.AgentState;
            var img = stateImage != null ? stateImage.GetComponent<Image>() : null;
            if (img != null)
            {
                if (currentState == "White") img.color = Color.white;
                else if (currentState == "Green") img.color = Color.green;
                else if (currentState == "Red") img.color = Color.red;
                else img.color = Color.white;
            }

            
            if (stateImage != null)
            {
                stateImage.onClick.AddListener(() =>
                {
                    if (currentState == "White") currentState = "Green";
                    else if (currentState == "Green") currentState = "Red";
                    else currentState = "White";

                    if (img != null)
                    {
                        if (currentState == "White") img.color = Color.white;
                        else if (currentState == "Green") img.color = Color.green;
                        else if (currentState == "Red") img.color = Color.red;
                    }

                    // unity_info 동기화: FileList의 unity_id 기준으로 상태 반영
                    string unityIdLocal = FileList.ContainsKey("unity_id") ? FileList["unity_id"]?.ToString() : string.Empty;
                    if (!string.IsNullOrEmpty(unityIdLocal))
                    {
                        PrefabInfo.UpdateImportedObjectUnityStateByUnityId(unityIdLocal, currentState);
                    }
                });
            }
            
            string unityId = FileList.ContainsKey($"unity_id") ? FileList[$"unity_id"]?.ToString() : "";
            // 생성 시 상태는 일단 그대로 두고, 일괄 동기화에서 결정
            DeleteButton.onClick.AddListener(() =>
            {
                // PrefabInfo.RemoveImportedObjectInfoByFileId(fileId);
                PrefabInfo.RemoveImportedObjectInfoByFileIdwithunityId(unityId,fileId);
                var _list = GameManager.scenarioEdit.scenario_agentListDict;
                int _idx = _list.FindIndex(dict =>
                    dict != null && dict.ContainsKey("agent_id") && dict["agent_id"]?.ToString() == fileId
                );
                if (_idx != -1)
                {
                    _list.RemoveAt(_idx); // 동일 id 하나만 제거
                }
                Destroy(newButtonObj);
                // 다음 프레임에 실제 파괴가 반영된 후 동기화
                StartCoroutine(SyncDeleteButtonsNextFrame());
            });
            
            // 새로운 content가 추가된 후 즉시 동기화
            SyncAllocateDeleteButtons();
        } 
    }

    // 콘텐츠 개수 규칙에 따른 delete 버튼 상태 동기화
    private void SyncAllocateDeleteButtons()
    {
        var content = GameManager.scenarioEdit.scinfo.scenarioEditInfo.AllocateObjectContent.transform;
        int childCount = content.childCount;

        for (int i = 0; i < childCount; i++)
        {
            var child = content.GetChild(i);
            var delButton = child.Find("Del")?.GetComponent<Button>();
            if (childCount <= 2)
            {
                // 두 개일 때는 두 개 모두 비활성화
                delButton.interactable = false;
            }
            else // childCount >= 3
            {
                // 세 개 이상일 때는 첫 번째만 비활성화, 나머지는 활성화
                delButton.interactable = (i != 0);
            }
        }
    }

    private IEnumerator SyncDeleteButtonsNextFrame()
    {
        yield return null; // Destroy 반영 대기
        SyncAllocateDeleteButtons();
    }

    public void UpdateScenarioUI_DeplyablePlatforms()
    {
        StartCoroutine(WaitForCommResultCoroutine("Agent"));
    }

    IEnumerator WaitForCommResultCoroutine(string commDB)
    {
        yield return new WaitForSeconds(1.5f);
        GameManager.communication.Communication(commDB, new List<string> { "id", "name", "description" }, new Dictionary<string, object>(), "POST" );
        yield return new WaitForSeconds(1.5f);

        ObjectFiletoButton(commDB, GameManager.communication.Url_result);
        
    }

    void ObjectFiletoButton(string table, string FileList)
    {
        // Green, Red, White PlatformContent의 모든 자식 오브젝트 삭제
        var greenContent = GameManager.scenarioEdit.scinfo.scenarioEditInfo.GreenPlatformContent.transform;
        var redContent = GameManager.scenarioEdit.scinfo.scenarioEditInfo.RedPlatformContent.transform;
        var whiteContent = GameManager.scenarioEdit.scinfo.scenarioEditInfo.WhitePlatformContent.transform;

        foreach (Transform child in greenContent)
        {
            Destroy(child.gameObject);
        }
        foreach (Transform child in redContent)
        {
            Destroy(child.gameObject);
        }
        foreach (Transform child in whiteContent)
        {
            Destroy(child.gameObject);
        }

        // FileList가 null이거나 빈 문자열이면 리턴
        if (string.IsNullOrEmpty(FileList))
        {
            return;
        }

        var FileListJArray = Newtonsoft.Json.Linq.JArray.Parse(FileList);

        foreach (var fileObj in FileListJArray)
        {
            // fileObj는 JObject임
            var file = fileObj as Newtonsoft.Json.Linq.JObject;
            if (file == null) continue;

            string fileName = file["name"]?.ToString();
            string fileId = file["id"]?.ToString();
            string fileDesc = file["description"]?.ToString();

            // green, red, white 모두 동일하게 버튼 생성
            Transform[] parentTransforms = new Transform[] { greenContent, redContent, whiteContent };
            foreach (var parentTransform in parentTransforms)
            {
                GameObject newButtonObj = Instantiate(
                    GameManager.scenarioEdit.scinfo.scenarioEditInfo.deployableButton, 
                    parentTransform
                );
                Button newButton = newButtonObj.GetComponent<Button>();

                Text buttonText = newButton.GetComponentInChildren<Text>();
                if (buttonText != null)
                {
                    buttonText.text = fileName;
                }
                newButton.onClick.RemoveAllListeners();

                // parentTransform에 따라 AgentState를 다르게 설정
                if (parentTransform == greenContent)
                {
                    newButton.onClick.AddListener(() => {
                        GameManager.scenarioEdit.AgentState = "Green";
                        GameManager.scenarioEdit.ScenarioAgentButton(fileId, fileName, fileDesc, 
                            GameManager.scenarioEdit.ScenarioObject.transform, table);
                    });
                }
                else if (parentTransform == redContent)
                {
                    newButton.onClick.AddListener(() => {
                        GameManager.scenarioEdit.AgentState = "Red";
                        GameManager.scenarioEdit.ScenarioAgentButton(fileId, fileName, fileDesc, 
                            GameManager.scenarioEdit.ScenarioObject.transform, table);
                    });
                }
                else if (parentTransform == whiteContent)
                {
                    newButton.onClick.AddListener(() => {
                        GameManager.scenarioEdit.AgentState = "White";
                        GameManager.scenarioEdit.ScenarioAgentButton(fileId, fileName, fileDesc, 
                            GameManager.scenarioEdit.ScenarioObject.transform, table);
                    });
                }
            }
        }
    }

    public void ImportScenarioToCreate(string fileId, string fileName, string fileDesc, Transform Position, Transform Parent, string table)
    {
        // Parent 오브젝트의 모든 자식 오브젝트 삭제
        if (Parent != null)
        {
            for (int i = Parent.childCount - 1; i >= 0; i--)
            {
                GameObject.Destroy(Parent.GetChild(i).gameObject);
            }
        }

        GameManager.scenarioEdit.ImportScenario(fileId, fileName, fileDesc, Position, Parent, table);
    }
}
