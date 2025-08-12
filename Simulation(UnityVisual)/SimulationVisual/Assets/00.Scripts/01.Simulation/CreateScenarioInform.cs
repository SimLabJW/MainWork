using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System; // [Serializable]을 사용하려면 System 네임스페이스가 필요합니다.
using Button = UnityEngine.UI.Button;

public class CreateScenarioInform : MonoBehaviour
{
    [Header("CreateScenario Content")]
    [SerializeField]
    public CreateScenarioInfo createScenarioInfo = new CreateScenarioInfo();

    [Serializable]
    public class CreateScenarioInfo
    {

        [Header("Simulation Env")]
        public Transform Simulation_ENV;

        [Header("Editor Content")]
        public EditorInform editorInform = new EditorInform();

        [Header("Editor State")]
        public RawImage Env_rawimage;
        public RawImage Agent_rawimage;

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

        [Header("Select Object")]
        public GameObject Editor_ScrollView_Content;

        [Header("Map View")]
        public GameObject Editor_MapView;

        // Scenario Save Inform
        [Header("Scenario Save Inform")]
        public InputField Scenario_Name;
        public InputField Scenario_Description;
        public Button Scenario_Save;

    }

    [Header("Camera Type")]
    [SerializeField]
    public CreateScenarioCameraType cameraType = new CreateScenarioCameraType();

    [Serializable]
    public class CreateScenarioCameraType
    {
        public Camera MapView_Editor;
    }
}
