using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using UnityEngine.UI; // Text, Button 등 UI 컴포넌트 사용을 위해 추가
using UnityEngine.SceneManagement;

public class CreateScenario : MonoBehaviour
{
    [Header("Apply SimulationInform")]
    public CreateScenarioInform createScenarioInform;

    private CreateScenarioInform.CreateScenarioInfo csminfo;
    private CreateScenarioInform.CreateScenarioCameraType caminfo;

    void Start()
    {
        Debug.Log("CreateScenario Start");
        //Update Code to
        GameManager.createScenario.csminfo = createScenarioInform;   

        csminfo = GameManager.createScenario.csminfo.createScenarioInfo;
        caminfo = GameManager.createScenario.csminfo.cameraType;

        // add Listener to Button for create object list (Editor)
        csminfo.editorInform.Env_Button.onClick.AddListener(
            () => OnButtonclickEditor(csminfo.editorInform.Env_Button.GetComponentInChildren<Text>().text));
        csminfo.editorInform.Agent_Button.onClick.AddListener(
            () => OnButtonclickEditor(csminfo.editorInform.Agent_Button.GetComponentInChildren<Text>().text));

        // Scenario Inform
        csminfo.editorInform.Scenario_Name.onValueChanged.AddListener(
            value => GameManager.createScenario.SaveScenarioName = value
        );
        csminfo.editorInform.Scenario_Description.onValueChanged.AddListener(
            value => GameManager.createScenario.SaveScenarioDescription = value
        );
        csminfo.editorInform.Scenario_Save.onClick.AddListener(
            () => OnButtonclickEditor(csminfo.editorInform.Scenario_Save.GetComponentInChildren<Text>().text));
    }

    void OnButtonclickEditor(string buttonType)
    {
        switch (buttonType) 
        {
            case "Environment":
                Debug.Log("Env Button Clicked");
                StartCoroutine(WaitForCommResultCoroutine("Terrian"));
                break;

            case "Agent":
                Debug.Log("Agent Button Clicked");
                StartCoroutine(WaitForCommResultCoroutine("Agent"));
                break;

            case "Save":
                // PrefabInfo의 모든 import된 오브젝트 정보를 로그로 출력
                // PrefabInfo.LogAllImportedObjects();
                
                string ScenarioName = GameManager.createScenario.SaveScenarioName;
                string ScenarioDescription = GameManager.createScenario.SaveScenarioDescription;
               
                // List<string>을 string[] 배열로 변환
                string[] terrianFileIds = PrefabInfo.GetFileIdsByTable("Terrian").ToArray();
                string[] agentFileIds = PrefabInfo.GetFileIdsByTable("Agent").ToArray();

                // // 시나리오 정보에 오브젝트 데이터 추가
                // Debug.Log($"=== 시나리오 저장 정보 ===");
                // Debug.Log($"시나리오명: {ScenarioName}");
                // Debug.Log($"시나리오 설명: {ScenarioDescription}");

                // if (terrianFileIds.Length > 0)
                // {
                //     // 배열 자체를 바로 출력 (ToString()은 타입명만 나오므로, string.Join 사용 없이 배열 전체를 출력)
                //     Debug.Log($"환경 파일 ID 목록: {Newtonsoft.Json.JsonConvert.SerializeObject(terrianFileIds)}");
                // }

                // if (agentFileIds.Length > 0)
                // {
                //     Debug.Log($"에이전트 파일 ID 목록: {Newtonsoft.Json.JsonConvert.SerializeObject(agentFileIds)}");
                // }

                StartCoroutine(WaitForCommResultSaveCoroutine(ScenarioName, ScenarioDescription, terrianFileIds, agentFileIds));
                
                break;
        }
    }

    void jsonFiletoButton(string table, string FileList)
    {
        foreach (Transform child in csminfo.editorInform.Editor_ScrollView_Content.transform)
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

            Button newButton = Instantiate(csminfo.editorInform.Editor_Button, 
                csminfo.editorInform.Editor_ScrollView_Content.transform);

            string fileName = file["name"]?.ToString();
            string fileId = file["id"]?.ToString();

            Text buttonText = newButton.GetComponentInChildren<Text>();
            if (buttonText != null)
            {
                buttonText.text = fileName;
            }
            newButton.onClick.RemoveAllListeners();

            newButton.onClick.AddListener(() => ClickedForInstantiate(table, fileId, fileName));
        }
    }

    void ClickedForInstantiate(string table, string fileId, string fileName)
    {
      
        switch (table)
        {
            case "Terrian":
                csminfo.Env_rawimage.color = Color.green;
                GameManager.createScenario.ImportObject(fileId, fileName,
                    csminfo.Simulation_ENV, csminfo.Simulation_ENV, table);
                caminfo.MapView_Editor.gameObject.SetActive(true);

                GameManager.createScenario.Editor_ENV = true;
                csminfo.editorInform.Agent_Button.interactable = true;
                //Fit Camera To Rawimage
                StartCoroutine(DelayedFit());
                

                break;
            case "Agent":
                if (GameManager.createScenario.Editor_ENV)
                {
                    csminfo.Agent_rawimage.color = Color.green;
                    GameManager.createScenario.EditorViewControl(fileId, fileName,
                    csminfo.Simulation_ENV, table);
                }
                break;

        }
    }

    IEnumerator DelayedFit()
    {
        yield return new WaitForSeconds(1.58f);
        Debug.Log("fit camera");
        GameManager.createScenario.AddEditorViewFit(csminfo.Simulation_ENV.gameObject);
    }
    
    IEnumerator WaitForCommResultCoroutine(string commDB)
    {
        GameManager.communication.Communication(commDB, new List<string> { "id", "name" }, new Dictionary<string, object>(), "POST" );
        yield return new WaitForSeconds(1.5f);

        jsonFiletoButton(commDB, GameManager.communication.Url_result);
        
    }

    IEnumerator WaitForCommResultSaveCoroutine(string scenarioName, string scenarioDescription, string[] terrianList, string[] agentList)
    {
        GameManager.createScenario.SaveScenario(scenarioName, scenarioDescription, terrianList, agentList);
        
        yield return new WaitForSeconds(1.5f);

        SceneManager.LoadScene("ScenarioEditorScene");
        
    }
}
