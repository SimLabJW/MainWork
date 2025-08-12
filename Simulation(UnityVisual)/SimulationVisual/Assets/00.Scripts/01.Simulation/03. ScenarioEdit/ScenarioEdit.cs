using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;
using UnityEngine.UI;
using UnityEngine;
using UnityEngine.SceneManagement;

public class ScenarioEdit : MonoBehaviour
{
    [Header("Apply ScenarioEditInform")]
    public ScenarioEditInform scenarioEditInform;

    public GameObject FileListUI;
    private ScenarioEditInform.ScenarioEditInfo scinfo;

    void Awake()
    {
        GameManager.scenarioEdit.scinfo = scenarioEditInform;
        scinfo = GameManager.scenarioEdit.scinfo.scenarioEditInfo;
    }

    void Start()
    {
        // scenario buttons 
        scinfo.scenarioEdit.NewScenario.onClick.AddListener(
            () => OnButtonclickEditor(scinfo.scenarioEdit.NewScenario.GetComponentInChildren<Text>().text)
        );
        scinfo.scenarioEdit.OpenScenario.onClick.AddListener(
            () => OnButtonclickEditor(scinfo.scenarioEdit.OpenScenario.GetComponentInChildren<Text>().text)
        );

        // environment edit button
        scinfo.environmentEdit.EnvironmentEditButton.onClick.AddListener(
            () => {
                var canvasGroup = scinfo.environmentEdit.EnvironmentPanel.GetComponent<CanvasGroup>();
                if (canvasGroup != null)
                {
                    canvasGroup.interactable = true;
                }
            }
        );
    }

    void OnButtonclickEditor(string buttonType)
    {
        var Path = string.Empty;
        switch (buttonType) 
        {
            case "New Scenario":
                SceneManager.LoadScene("CreateScenarioDefaultScene");
                break;

            case "Open Scenario":
                FileListUI.SetActive(true);
                break;

            case "Save Scenario":
                break;

            case "Delete Scenario":
                break;
        }
    }

}
