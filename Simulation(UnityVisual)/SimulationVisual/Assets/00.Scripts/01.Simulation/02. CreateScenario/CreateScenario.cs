using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using UnityEngine.UI; // Text, Button 등 UI 컴포넌트 사용을 위해 추가
using UnityEngine.SceneManagement;
using System.Threading.Tasks;

public class CreateScenario : MonoBehaviour
{
    [Header("Apply SimulationInform")]
    public CreateScenarioInform createScenarioInform;

    private CreateScenarioInform.CreateScenarioInfo csminfo;
    private CreateScenarioInform.CreateScenarioCameraType caminfo;

    public GameObject ScenarioEditObject;

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

                string ScenarioName = GameManager.createScenario.SaveScenarioName;
                string ScenarioDescription = GameManager.createScenario.SaveScenarioDescription;

                string[] terrianFileIds = PrefabInfo.GetFileIdsByTable("Terrian").ToArray();
                string[] agentFileIds = PrefabInfo.GetFileIdsByTable("Agent").ToArray();

                StartCoroutine(WaitForCommResultSaveCoroutine(ScenarioName, ScenarioDescription, terrianFileIds, agentFileIds));
                caminfo.MapView_Editor.gameObject.SetActive(false);

                gameObject.SetActive(false);
                ScenarioEditObject.SetActive(true);

                
                caminfo.MapView_Editor.gameObject.SetActive(true);
                // caminfo.MapView_Editor.gameObject.SetActive(false);

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
            string fileDesc = file["desc"]?.ToString();

            Text buttonText = newButton.GetComponentInChildren<Text>();
            if (buttonText != null)
            {
                buttonText.text = fileName;
            }
            newButton.onClick.RemoveAllListeners();

            newButton.onClick.AddListener(() => ClickedForInstantiate(table, fileId, fileName, fileDesc));
        }
    }

    void ClickedForInstantiate(string table, string fileId, string fileName, string fileDesc)
    {
      
        switch (table)
        {
            case "Terrian":
                csminfo.Env_rawimage.color = Color.green;
                GameManager.createScenario.ImportObject(fileId, fileName, fileDesc,
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
                    GameManager.createScenario.EditorViewControl(fileId, fileName, fileDesc,
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
