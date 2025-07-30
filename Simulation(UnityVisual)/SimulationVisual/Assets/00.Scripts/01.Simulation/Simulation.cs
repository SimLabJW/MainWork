using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
public class Simulation : MonoBehaviour
{
    [Header("Apply SimulationInform")]
    public SimulationInform simulationInform;

    private SimulationInform.SimulationInfo sminfo;
    private SimulationInform.SimulationCameraType caminfo;

    private ObjectProfile profile = new ObjectProfile();

    void Start()
    {
        //Update Code to
        GameManager.simulation.sm = simulationInform;   

        sminfo = GameManager.simulation.sm.simulationInfo;
        caminfo = GameManager.simulation.sm.cameraType;

        // add Listener to Button for create object list (Editor)
        sminfo.editorInform.Env_Button.onClick.AddListener(
            () => OnButtonclickEditor(sminfo.editorInform.Env_Button.GetComponentInChildren<Text>().text));
        sminfo.editorInform.Agent_Button.onClick.AddListener(
            () => OnButtonclickEditor(sminfo.editorInform.Agent_Button.GetComponentInChildren<Text>().text));
        sminfo.editorInform.Scenario_Button.onClick.AddListener(
            () => OnButtonclickEditor(sminfo.editorInform.Scenario_Button.GetComponentInChildren<Text>().text));
        
        sminfo.editorInform.env_profile.EnvDropDown.onValueChanged.AddListener(OnEnvDropdownChanged);
        sminfo.editorInform.agent_profile.AgentDropDown.onValueChanged.AddListener(OnAgentDropdownChanged);

    }
    
    void OnButtonclickEditor(string buttonType)
    {
        var Path = string.Empty;
        switch (buttonType) 
        {
            case "Env":
                Path = "C:/Users/USER/Desktop/HDRP_Folder/Create/Map/";
                GameManager.simulation.sm.simulationInfo.editorInform.env_profile.Env_Profile.SetActive(true);
                GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.Agent_Profile.SetActive(false);
                GameManager.simulation.sm.simulationInfo.editorInform.scenario_profile.Scenario_Profile.SetActive(false);
                jsonFiletoButton("env",Path);
                break;

            case "Agent":
                Path = "C:/Users/USER/Desktop/HDRP_Folder/Create/Agent/";
                GameManager.simulation.sm.simulationInfo.editorInform.env_profile.Env_Profile.SetActive(false);
                GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.Agent_Profile.SetActive(true);
                GameManager.simulation.sm.simulationInfo.editorInform.scenario_profile.Scenario_Profile.SetActive(false);
                jsonFiletoButton("agent",Path);
                break;

            case "Scenario":
                Path = "C:/Users/USER/Desktop/HDRP_Folder/Create/Scenario/";
                GameManager.simulation.sm.simulationInfo.editorInform.env_profile.Env_Profile.SetActive(false);
                GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.Agent_Profile.SetActive(false);
                GameManager.simulation.sm.simulationInfo.editorInform.scenario_profile.Scenario_Profile.SetActive(true);
                jsonFiletoButton("scenario", Path);
                break;

            case "Save":
                GameManager.simulation.SaveObejct = true;
                break;
        }
    }

    void jsonFiletoButton(string mode,string path)
    {
        foreach (Transform child in sminfo.editorInform.Editor_ScrollView_Content.transform)
        {
            Destroy(child.gameObject);
        }


        if (!Directory.Exists(path))
        {
            return;
        }

        string[] files = Directory.GetFiles(path, "*.glb");
        
        foreach (string file in files)
        {
            Button newButton = Instantiate(sminfo.editorInform.Editor_Button, 
                sminfo.editorInform.Editor_ScrollView_Content.transform);

            string fileName = Path.GetFileNameWithoutExtension(file);

            Text buttonText = newButton.GetComponentInChildren<Text>();
            if (buttonText != null)
            {
                buttonText.text = fileName;
            }
            newButton.onClick.RemoveAllListeners();

            newButton.onClick.AddListener(() => ClickedForInstantiate(mode, path, fileName));
        }
    }

    void ClickedForInstantiate(string mode, string path, string fileName)
    {
      
        switch (mode)
        {
            case "env":
                sminfo.Env_rawimage.color = Color.green;
                GameManager.simulation.ImportObject(path, fileName,
                    sminfo.Simulation_ENV, sminfo.Simulation_ENV, "env");
                caminfo.MapView_Editor.gameObject.SetActive(true);

                GameManager.simulation.Editor_ENV = true;
                //Fit Camera To Rawimage
                StartCoroutine(DelayedFit());
                

                break;
            case "agent":
                if (GameManager.simulation.Editor_ENV)
                {
                    sminfo.Agent_rawimage.color = Color.green;
                    GameManager.simulation.EditorViewControl(path, fileName,
                    sminfo.Simulation_ENV);
                }
                break;

            case "scenario":
                sminfo.Scenario_rawimage.color = Color.green;
                break;

        }
    }

    IEnumerator DelayedFit()
    {
        yield return new WaitForSeconds(0.2f);
        GameManager.simulation.AddEditorViewFit(sminfo.Simulation_ENV.gameObject);
    }

    void InstantiateDropDownButton(List<GameObject> profile, GameObject View)
    {
        foreach (Transform child in View.transform)
        {
            Destroy(child.gameObject);
        }

        foreach (GameObject Object in profile)
        {
            Button newButton = Instantiate(sminfo.editorInform.Editor_Button,
                View.transform);

            Text buttonText = newButton.GetComponentInChildren<Text>();
            if (buttonText != null)
            {
                buttonText.text = Object.name;
            }

        }
    }
    void OnEnvDropdownChanged(int index)
    {
        string selected = sminfo.editorInform.env_profile.EnvDropDown.options[index].text;

        if (selected == "Planes")
        {
            var env_profile = profile.MapProfileButtonApply(GameManager.simulation.currentObeject);
            InstantiateDropDownButton(env_profile.PlaneObject, 
                GameManager.simulation.sm.simulationInfo.editorInform.env_profile.EnvScrollView);
        }
        else if (selected == "Obstacles")
        {
            var env_profile = profile.MapProfileButtonApply(GameManager.simulation.currentObeject);
            InstantiateDropDownButton(env_profile.ObstacleObject, 
                GameManager.simulation.sm.simulationInfo.editorInform.env_profile.EnvScrollView);
        }
        else 
        {
            foreach (Transform child in GameManager.simulation.sm.simulationInfo.editorInform.env_profile.EnvScrollView.transform)
            {
                Destroy(child.gameObject);
            }
        }

        
    }
    void OnAgentDropdownChanged(int index)
    {
        string selected = sminfo.editorInform.agent_profile.AgentDropDown.options[index].text;

        if (selected == "Wheels")
        {
            var agent_profile = profile.AgentProfileButtonApply(GameManager.simulation.currentObeject);
            InstantiateDropDownButton(agent_profile.WheelObject,
                GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.AgentScrollView);
        }
        else if (selected == "Floaters")
        {
            var agent_profile = profile.AgentProfileButtonApply(GameManager.simulation.currentObeject);
            InstantiateDropDownButton(agent_profile.FloaterObject,
                GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.AgentScrollView);
        }
        else if (selected == "Sensors")
        {
            var agent_profile = profile.AgentProfileButtonApply(GameManager.simulation.currentObeject);
            InstantiateDropDownButton(agent_profile.SensorObject,
                GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.AgentScrollView);
        }
        else if (selected == "Weapones")
        {
            var agent_profile = profile.AgentProfileButtonApply(GameManager.simulation.currentObeject);
            InstantiateDropDownButton(agent_profile.WeaponeObject,
                GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.AgentScrollView);
        }
        else
        {
            foreach (Transform child in
                GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.AgentScrollView.transform)
            {
                Destroy(child.gameObject);
            }
        }
    }

}
