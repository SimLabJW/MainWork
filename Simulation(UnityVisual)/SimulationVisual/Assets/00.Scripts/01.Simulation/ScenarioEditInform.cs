using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using UnityEngine.UI;
public class ScenarioEditInform : MonoBehaviour
{
    [Header("ScenarioEdit Content")]
    [SerializeField]
    public ScenarioEditInfo scenarioEditInfo = new ScenarioEditInfo();

    [Serializable]
    public class ScenarioEditInfo
    {
        [Header("Simulation Env")]
        public Transform Simulation_ENV;
        public GameObject Agent_Size;

        [Header("Scenario Set")]
        public ScenarioSet scenarioSet = new ScenarioSet();

        [Header("Environment Set")]
        public EnvironmentEdit environmentEdit= new EnvironmentEdit();

        [Header("DeployablePlatform Set")]
        public GameObject GreenPlatformContent;
        public GameObject RedPlatformContent;
        public GameObject WhitePlatformContent;
        public GameObject deployableButton;

        [Header("AllocateObject Set")]
        public GameObject AllocateObjectContent;
        public GameObject AllocateObjectButton;

        [Header("Waypoint Set")]
        public GameObject waypointContent;
        public GameObject WaypointObjectButton;
        public GameObject WaypointViewPrefab;
        public GameObject WaypointConnectPrefab;
    }

    [Serializable]
    public class ScenarioSet
    {
        [Header("Scenario Info")]
        public Button NewScenario;
        public Button OpenScenario;
        public Button SaveScenario;
        public Button DeleteScenario;

        public Text ScenarioName;
        public Text ScenarioDescription;
    }

    [Serializable]
    public class EnvironmentEdit
    {
        [Header("Envrionment Panel")]
        public GameObject EnvironmentPanel;
        public Button EnvironmentEditButton;

        [Header("Environment Value")]
        public EnvironmentValue environmentvalue = new EnvironmentValue();
    }

    [Serializable]
    public class EnvironmentValue
    {
        public GameObject WeatherType; // 맑음/비/눈/번개 등 날씨

        public Text LightingIntensity_T; // 광원 세기
        public Slider LightingIntensity_S;

        public Text SunAngle_T; // 태양의 위치
        public Slider SunAngle_S;

        public Text Temperature_T; // 온도
        public Slider Temperature_S;

        public Text RainIntensity_T; // 비의 세기
        public Slider RainIntensity_S;

        public Text Visibility_T; // 시야 거리
        public Slider Visibility_S;

        public Text WaveHeight_T; //파도 높이
        public Slider WaveHeight_S;

        public Text WaveSpeed_T; // 파도 속도
        public Slider WaveSpeed_S;

        public Slider WaveDirectionNS; // 파도 진행 방향
        public Slider WaveDirectionWE;

        public Text WaveClarity_T; // 수중 가시 거리
        public Slider WaveClarity_S;

        public Text BuoyancyStrength_T; // 부력 영향 정도
        public Slider BuoyancyStrength_S;

        public Text SeaLevel_T; // 해수면 높이
        public Slider SeaLevel_S;
    }

    [Header("Camera Type")]
    [SerializeField]
    public ScenarioEditCameraType cameraType = new ScenarioEditCameraType();

    [Serializable]
    public class ScenarioEditCameraType
    {
        [Header("Map View")]
        public GameObject Editor_ScenarioView;
        [Header("Scenario Camera")]
        public Camera ScenarioView_Editor;
    }

}
