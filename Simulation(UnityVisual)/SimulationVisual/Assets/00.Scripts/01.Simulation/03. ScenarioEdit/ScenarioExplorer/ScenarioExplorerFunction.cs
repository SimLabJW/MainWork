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
    public Text SelectScenarioText;
    private bool SelectScenario = false;

    [Space(10)]
    [Header("Scenario Explorer ButtonListContent")]
    public GameObject ScenarioContent;


    // Start is called before the first frame update
    void Start()
    {
        ReturnButton.onClick.AddListener(ReturnButtonClick);
        CheckButton.onClick.AddListener(CheckButtonClick);
    }

    void ReturnButtonClick()
    {
        FileListUI.SetActive(false);
    }

    void CheckButtonClick()
    {
        if(SelectScenario)
        {
            FileListUI.SetActive(false);
        }
        
    }
    
    void AddScenarioRow(int index, string name, string description)
    {
        // // Row 프리팹 생성
        // GameObject newRow = Instantiate(rowPrefab, contentParent);

        // // 버튼(행) 아래에 존재하는 텍스트 오브젝트들 찾기
        // Text indexText = newRow.transform.Find("Index_T").GetComponent<Text>();
        // Text nameText = newRow.transform.Find("Name_T").GetComponent<Text>();
        // Text descriptionText = newRow.transform.Find("Description_T").GetComponent<Text>();

        // // 텍스트 설정
        // indexText.text = index.ToString();
        // nameText.text = name;
        // descriptionText.text = description;
    }



}
