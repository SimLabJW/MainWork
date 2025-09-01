using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ScenarioExplorerFunction : MonoBehaviour
{
    [Header("Scenario Explorer")]
    public GameObject FileListUI;

    [Space(10)]
    [Header("Scenario Explorer Buttons")]
    public Button ReturnButton;
    public Button CheckButton;
 
    [Space(10)]
    [Header("Scenario Explorer Select")]
    // Text에서 InputField로 변경
    public InputField SelectScenarioInputField;
    private bool SelectScenario = false;

    [Space(10)]
    [Header("Scenario Explorer ButtonListContent")]
    public GameObject ScenarioContent;
    public GameObject buttonPrefab; // RowPrefab 프리팹

    private int Count = 1;

    void Start()
    {
        ReturnButton.onClick.AddListener(ReturnButtonClick);
        CheckButton.onClick.AddListener(CheckButtonClick);

        GameManager.scenarioEdit.CreateScenarioButtonAction -= AddScenarioRow;
        GameManager.scenarioEdit.CreateScenarioButtonAction += AddScenarioRow;
    }

    void ReturnButtonClick()
    {
        Count = 1;
        FileListUI.SetActive(false);
    }

    void CheckButtonClick()
    {
        if(SelectScenario)
        {
            GameManager.scenarioEdit.ClassifyScenarioInfo(GameManager.scenarioEdit.scenario_info);
            FileListUI.SetActive(false);
        }
    }
    
    void AddScenarioRow()
    {
        StartCoroutine(WaitForCommResultCoroutine("Scenario"));
    }
    
    IEnumerator WaitForCommResultCoroutine(string commDB)
    {
        GameManager.communication.Communication(commDB, new List<string> { "id", "name", "description" }, new Dictionary<string, object>(), "POST" );
        yield return new WaitForSeconds(1f);

        jsonFiletoButton(commDB, GameManager.communication.Url_result);
    }

    void jsonFiletoButton(string table, string FileList)
    {
        foreach (Transform child in ScenarioContent.transform)
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

            // buttonPrefab은 GameObject 타입이므로 Instantiate 후 Button 컴포넌트를 GetComponent로 가져와야 함
            GameObject newButtonObj = Instantiate(buttonPrefab, ScenarioContent.transform);
            Button newButton = newButtonObj.GetComponent<Button>();

            Text indexText = newButtonObj.transform.Find("Index_T").GetComponent<Text>();
            Text nameText = newButtonObj.transform.Find("Name_T").GetComponent<Text>();
            Text descriptionText = newButtonObj.transform.Find("Description_T").GetComponent<Text>();

            string fileId = file["id"]?.ToString();
            string fileName = file["name"]?.ToString();
            string fileDesc = file["description"]?.ToString();

            // 텍스트 설정
            indexText.text = Count.ToString();
            nameText.text = fileName;
            descriptionText.text = fileDesc;

            Count ++;

            if (newButton != null)
            {
                newButton.onClick.RemoveAllListeners();
                newButton.onClick.AddListener(() => GameManager.communication.ScenarioButtonCommunication(fileId));
                newButton.onClick.AddListener(() => SelectScenarioText(fileId, fileName));
                
            }
        }
    }

    public void SelectScenarioText(string fileID, string name)
    {
        // InputField의 text 값을 변경
        if (SelectScenarioInputField != null)
        {
            SelectScenarioInputField.text = name;
            GameManager.scenarioEdit.ScenarioId = fileID;
            Count = 1;
        }
        else
        {
            Debug.LogWarning("SelectScenarioInputField가 할당되지 않았습니다.");
        }
        SelectScenario = true;
    }
}