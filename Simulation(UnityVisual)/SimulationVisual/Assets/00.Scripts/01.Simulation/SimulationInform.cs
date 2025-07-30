using System;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.UIElements;
using Button = UnityEngine.UI.Button;
using Toggle = UnityEngine.UI.Toggle;

public class SimulationInform : MonoBehaviour
{
    [Header("Simulation Content")]
    [SerializeField]
    public SimulationInfo simulationInfo = new SimulationInfo(); // 초기화 필요

    [Serializable]
    public class SimulationInfo
    {

        [Header("Simulation Env")]
        public Transform Simulation_ENV;

        [Header("Editor Content")]
        public EditorInform editorInform = new EditorInform();

        [Header("Visual Content")]
        public VisualInform visualInform = new VisualInform();

        [Header("Editor State")]
        public RawImage Env_rawimage;
        public RawImage Agent_rawimage;
        public RawImage Scenario_rawimage;


    }

    [Serializable]
    public class EditorInform
    {
        [Header("Button Object")]
        public Button Editor_Button;
        public GameObject Agent_Size;
        public GameObject Scenario_Point;

        [Header("Select Object Type")]
        public Button Env_Button;
        public Button Agent_Button;
        public Button Scenario_Button;

        [Header("Select Object")]
        public GameObject Editor_ScrollView_Content;

        [Header("Map View")]
        public GameObject Editor_MapView;

        [Header("Profile Object")]
        public Text Object_type;
        [Space(3)]
        public EnvProfile env_profile;
        [Space(3)]
        public AgentProfile agent_profile;
        [Space(3)]
        public ScenarioProfile scenario_profile;
        [Space(3)]

        

        [Header("Next Phase")]
        public GameObject Done_Button;
        public GameObject Next_Phase;
    }

    [Serializable]
    public class EnvProfile
    {
        [Header("Env Object")]
        public GameObject Env_Profile;

        [Header("Env Inform")]
        public InputField EnvName;
        public InputField Env_x;
        public InputField Env_y;
        public InputField Env_z;

        [Header("Env Object View")]
        public Dropdown EnvDropDown;
        public GameObject EnvScrollView;
    }

    [Serializable]
    public class AgentProfile
    {
        [Header("Agent Object")]
        public GameObject Agent_Profile;

        [Header("Agent Inform")]
        public InputField AgentName;
        public InputField Agent_x;
        public InputField Agent_y;
        public InputField Agent_z;

        [Header("Agent Object View")]
        public Dropdown AgentDropDown;
        public GameObject AgentScrollView;

    }

    [Serializable]
    public class ScenarioProfile
    {
        [Header("Scenario Object")]
        public GameObject Scenario_Profile;

        [Header("Syncro Scenario Inform")]
        public ScenarioSyncro scenarioSyncro;

        [Header("Scenario Create")]
        public ScenarioCreate scenarioCreate;


    }

    [Serializable]
    public class ScenarioSyncro
    {
        [Header("Scenario Inform")]
        public InputField scenarioTime;
        public InputField envName;
        public Dropdown composition_agent;
        public InputField location;
        public RawImage pathColor;
        public Dropdown simulationPriority;
        public InputField strategicObjective;

    }
    [Serializable]
    public class ScenarioCreate
    {
        [Header("Create Scenario")]
        public InputField agentName;
        public Dropdown selectStrategicObjective;
        public Button Point;
        public Button ReturnPoint;
        public Button DeleteAllPoint;
        public Toggle CameraSelect;
        public Button Save_Button;

    }

    [Serializable]
    public class VisualInform
    {
        [Header("Select Object Type")]
        public Dropdown Camera_View_Button;
        public GameObject Editor_ScrollView;

        [Header("Control Data Text")]
        public Text Control_data;

        [Header("MiniMap View")]
        public GameObject Visual_MiniMapView;

        [Header("Stop Phase")]
        public GameObject Done_Button;
    }

    [Header("Camera Type")]
    [SerializeField]
    public SimulationCameraType cameraType = new SimulationCameraType();

    [Serializable]
    public class SimulationCameraType
    {
        public Camera MapView_Editor;

        public Camera MiniMapView_Visual;
        public Camera PlayerView;
        public Camera ObjectView;
    }

}
