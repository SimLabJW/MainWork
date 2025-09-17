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

    public GameObject NewScenarioCreate;

    public Button StartSimulation;
    public GameObject StartSimulationUI;
    // 환경 설정 초기값을 저장할 딕셔너리
    private Dictionary<string, string> initialEnvironmentValues = new Dictionary<string, string>();
    private Dictionary<string, float> initialSliderValues = new Dictionary<string, float>();

    void Start()
    {
        GameManager.scenarioEdit.scinfo = scenarioEditInform;
        scinfo = GameManager.scenarioEdit.scinfo.scenarioEditInfo;
        
        // 시나리오 불러오기 전에 현재 환경 설정값 저장
        SaveInitialEnvironmentValues();
        
        // scenario buttons 
        scinfo.scenarioSet.NewScenario.onClick.AddListener(
            () => OnButtonclickEditor(scinfo.scenarioSet.NewScenario.GetComponentInChildren<Text>().text)
        );
        scinfo.scenarioSet.OpenScenario.onClick.AddListener(
            () => OnButtonclickEditor(scinfo.scenarioSet.OpenScenario.GetComponentInChildren<Text>().text)
        );
        scinfo.scenarioSet.SaveScenario.onClick.AddListener(
            () => OnButtonclickEditor(scinfo.scenarioSet.SaveScenario.GetComponentInChildren<Text>().text)
        );
        scinfo.scenarioSet.DeleteScenario.onClick.AddListener(
            () => OnButtonclickEditor(scinfo.scenarioSet.DeleteScenario.GetComponentInChildren<Text>().text)
        );


        // environment edit button
        scinfo.environmentEdit.EnvironmentEditButton.onClick.AddListener(
            () => {
                var canvasGroup = scinfo.environmentEdit.EnvironmentPanel.GetComponent<CanvasGroup>();
                if (canvasGroup != null)
                {
                    // interatable 상태를 토글
                    if (!canvasGroup.interactable)
                    {
                        settingEnvironmentEdit();
                        canvasGroup.interactable = true;
                    }
                    else
                    {
                        canvasGroup.interactable = false;
                    }
                }
            }
        );

        // 시작 시 비활성화 (ScenarioObject 없으므로)
        if (StartSimulation != null)
        {
            StartSimulation.interactable = false;
            StartSimulation.onClick.AddListener(OnClickStartSimulation);
        }
    }

    
    void OnButtonclickEditor(string buttonType)
    {
        var Path = string.Empty;
        switch (buttonType) 
        {
            case "New Scenario":
                // SceneManager.LoadScene("CreateScenarioDefaultScene");
                gameObject.SetActive(false);
                NewScenarioCreate.SetActive(true);
                break;

            case "Open Scenario":
                FileListUI.SetActive(true);
                StartSimulation.interactable = true;
                StartCoroutine(DelayedScenarioButton());
                break;

            case "Save Scenario": // db/update
                if(GameManager.scenarioEdit.ScenarioObject != null)
                {
                    var glb_filters = new Dictionary<string, object>
                    {
                        { "id", GameManager.scenarioEdit.ScenarioId }
                    };
                    
                    // 1.Scenario  glb  Update : 1.find glb id, 2. update glb
                    GameManager.communication.ScenarioInfoFind("Scenario", new List<string> { "glb_id" }, glb_filters);
                    
                    // 2. Scenario Agents List Insert&Delete( table : Scenario_agent, GameManager.scenarioEdit.scenario_agentListDict , filter : Scenario_id)
                    // 계산은 UrlComm.cs의 Scenario_Agent 응답 처리에서 수행됨
                    // 2.1. find "Scenario_Agent" for agent_id
                    var scenario_agent_filters = new Dictionary<string, object>
                    {
                        { "scenario_id", GameManager.scenarioEdit.ScenarioId }
                    };

                    GameManager.communication.ScenarioInfoFind("Scenario_Agent", new List<string> { "id", "agent_id" }, scenario_agent_filters);
                    
                    // 3.Scenario Environment Update {tabel : Environment, filter : env_id / using  CommunicationScenarioUpdateUrl }
                    // GameManager.scenarioEdit.scenario_info["env_id"]

                    var env_filters = new Dictionary<string, object>
                    {
                        { "id", GameManager.scenarioEdit.scenario_info["env_id"] }
                    };

                    var data = BuildEnvironmentUpdateData();
                    GameManager.communication.ScenarioUpdate("Environment", data, env_filters);
                    
                }
                break;

            case "Delete Scenario": // db/delete
                if(GameManager.scenarioEdit.ScenarioObject != null)
                {
                    PrefabInfo.LogAllImportedObjects();
                    Destroy(GameManager.scenarioEdit.ScenarioObject.gameObject);
                    RestoreInitialEnvironmentValues();

                    // allocate content도 모두 삭제
                    var allocateContent = GameManager.scenarioEdit.scinfo.scenarioEditInfo.AllocateObjectContent.transform;
                    foreach (Transform child in allocateContent)
                    {
                        Destroy(child.gameObject);
                    }

                    // deployable platforms content도 모두 삭제
                    var deployableContent = GameManager.scenarioEdit.scinfo.scenarioEditInfo.GreenPlatformContent.transform;
                    foreach (Transform child in deployableContent)
                    {
                        Destroy(child.gameObject);
                    }

                    // 시나리오 이름, 설명 초기화 ("-")
                    var scenarioSet = GameManager.scenarioEdit.scinfo.scenarioEditInfo.scenarioSet;
                    if (scenarioSet != null)
                    {
                        if (scenarioSet.ScenarioName != null)
                            scenarioSet.ScenarioName.text = "-";
                        if (scenarioSet.ScenarioDescription != null)
                            scenarioSet.ScenarioDescription.text = "-";
                    }

                    // delete 통신 보내기
                    var filters = new Dictionary<string, object>
                    {
                        { "id", GameManager.scenarioEdit.ScenarioId }
                    };
                    GameManager.communication.ScenarioDelete("Scenario", filters);
                }
                break;
        }
    }

    void OnClickStartSimulation()
    {
        if (GameManager.scenarioEdit.ScenarioObject == null) return;

        // 요구사항: gameObject false, NewScenarioCreate false, SimulationEdit true
        // 현재 클래스의 gameObject 비활성화
        gameObject.SetActive(false);

        // NewScenarioCreate 비활성화
        if (NewScenarioCreate != null)
            NewScenarioCreate.SetActive(false);

        
        StartSimulationUI.SetActive(true);
        

    }

    IEnumerator DelayedScenarioButton()
    {
        yield return new WaitForSeconds(0.5f);
        GameManager.scenarioEdit.CreateScenarioButton();
    }

    void settingEnvironmentEdit()
    {
        // 광원 세기
        var LightingIntensity_T = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.LightingIntensity_T;
        var LightingIntensity_S = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.LightingIntensity_S;

        // 슬라이더 값이 바뀔 때 텍스트 갱신
        LightingIntensity_S.onValueChanged.AddListener((float val) => {
            if (LightingIntensity_T != null)
                LightingIntensity_T.text = val.ToString("0.##");
                GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("lighting_intensity", val.ToString("0.##"));
        });

        // 태양의 위치
        var SunAngle_T = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.SunAngle_T;
        var SunAngle_S = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.SunAngle_S;
        SunAngle_S.onValueChanged.AddListener((float val) => {
            if (SunAngle_T != null)
                SunAngle_T.text = val.ToString("0.##");
                GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("sun_angle", val.ToString("0.##"));
                
        });

        // 온도
        var Temperature_T = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.Temperature_T;
        var Temperature_S = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.Temperature_S;
        Temperature_S.onValueChanged.AddListener((float val) => {
            if (Temperature_T != null)
                Temperature_T.text = val.ToString("0.##");
                GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("temperature", val.ToString("0.##"));
        });

        // 비의 세기
        var RainIntensity_T = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.RainIntensity_T;
        var RainIntensity_S = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.RainIntensity_S;
        RainIntensity_S.onValueChanged.AddListener((float val) => {
            if (RainIntensity_T != null)
                RainIntensity_T.text = val.ToString("0.##");
                // GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("lighting_intensity", val.ToString("0.##"));
        });

        // 시야 거리
        var Visibility_T = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.Visibility_T;
        var Visibility_S = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.Visibility_S;
        Visibility_S.onValueChanged.AddListener((float val) => {
            if (Visibility_T != null)
                Visibility_T.text = val.ToString("0.##");
                // GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("lighting_intensity", val.ToString("0.##"));
        });

        // 파도 높이
        var WaveHeight_T = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveHeight_T;
        var WaveHeight_S = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveHeight_S;
        WaveHeight_S.onValueChanged.AddListener((float val) => {
            if (WaveHeight_T != null)
                WaveHeight_T.text = val.ToString("0.##");
                GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_height", val.ToString("0.##"));
        });

        // 파도 속도
        var WaveSpeed_T = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveSpeed_T;
        var WaveSpeed_S = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveSpeed_S;
        WaveSpeed_S.onValueChanged.AddListener((float val) => {
            if (WaveSpeed_T != null)
                WaveSpeed_T.text = val.ToString("0.##");
                GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_speed", val.ToString("0.##"));
        });

        // 파도 진행 방향 NS
        var WaveDirectionNS = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveDirectionNS;
        WaveDirectionNS.onValueChanged.AddListener((float val) => {
            GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_direction_ns", val.ToString("0.##"));
        });
        // 파도 진행 방향 WE
        var WaveDirectionWE = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveDirectionWE;
        WaveDirectionWE.onValueChanged.AddListener((float val) => {
            GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_direction_we", val.ToString("0.##"));
        });

        // 수중 가시 거리
        var WaveClarity_T = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveClarity_T;
        var WaveClarity_S = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.WaveClarity_S;
        WaveClarity_S.onValueChanged.AddListener((float val) => {
            if (WaveClarity_T != null)
                WaveClarity_T.text = val.ToString("0.##");
                GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_clarity", val.ToString("0.##"));
        });

        // 부력 영향 정도
        var BuoyancyStrength_T = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.BuoyancyStrength_T;
        var BuoyancyStrength_S = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.BuoyancyStrength_S;
        BuoyancyStrength_S.onValueChanged.AddListener((float val) => {
            if (BuoyancyStrength_T != null)
                BuoyancyStrength_T.text = val.ToString("0.##");
                GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("buoyancy_strength", val.ToString("0.##"));
        });

        // 해수면 높이
        var SeaLevel_T = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.SeaLevel_T;
        var SeaLevel_S = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue.SeaLevel_S;
        SeaLevel_S.onValueChanged.AddListener((float val) => {
            if (SeaLevel_T != null)
                SeaLevel_T.text = val.ToString("0.##");
                GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("sea_level", val.ToString("0.##"));
        });
        
    }

    // Environment 업데이트용 데이터 생성
    Dictionary<string, object> BuildEnvironmentUpdateData()
    {
        var env = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue;

        // UI에서 현재 표시 중인 값들을 문자열로 수집 (서버 스키마에 맞춰 key 구성)
        var data = new Dictionary<string, object>
        {
            { "weather_type", env.WeatherType.GetComponent<Dropdown>().options[env.WeatherType.GetComponent<Dropdown>().value].text },
            { "lighting_intensity", env.LightingIntensity_T != null ? env.LightingIntensity_T.text : null },
            { "sun_angle", env.SunAngle_T != null ? env.SunAngle_T.text : null },
            { "temperature", env.Temperature_T != null ? env.Temperature_T.text : null },
            { "rain_intensity", env.RainIntensity_T != null ? env.RainIntensity_T.text : null },
            { "visibility", env.Visibility_T != null ? env.Visibility_T.text : null },
            { "wave_height", env.WaveHeight_T != null ? env.WaveHeight_T.text : null },
            { "wave_speed", env.WaveSpeed_T != null ? env.WaveSpeed_T.text : null },
            { "wave_direction_ns", env.WaveDirectionNS != null ? env.WaveDirectionNS.value.ToString("0.##") : null },
            { "wave_direction_we", env.WaveDirectionWE != null ? env.WaveDirectionWE.value.ToString("0.##") : null },
            { "wave_clarity", env.WaveClarity_T != null ? env.WaveClarity_T.text : null },
            { "buoyancy_strength", env.BuoyancyStrength_T != null ? env.BuoyancyStrength_T.text : null },
            { "sea_level", env.SeaLevel_T != null ? env.SeaLevel_T.text : null }
        };

        return data;
    }

    // 환경 설정의 초기값을 저장하는 함수
    void SaveInitialEnvironmentValues()
    {
        // Unity의 실제 환경 설정값을 가져와서 저장 (UI 값이 아닌)
        
        // 광원 세기 - Unity 기본값 18000
        initialEnvironmentValues["LightingIntensity"] = "-";
        initialSliderValues["LightingIntensity"] = 0;
        
        // 태양의 위치 - Unity 기본값 0도
        initialEnvironmentValues["SunAngle"] = "-";
        initialSliderValues["SunAngle"] = 0f;
        
        // 온도 - Unity 기본값 5500K
        initialEnvironmentValues["Temperature"] = "-";
        initialSliderValues["Temperature"] = 0;
        
        // 비의 세기 - Unity 기본값 0
        initialEnvironmentValues["RainIntensity"] = "-";
        initialSliderValues["RainIntensity"] = 0f;
        
        // 시야 거리 - Unity 기본값 0
        initialEnvironmentValues["Visibility"] = "-";
        initialSliderValues["Visibility"] = 0f;
        
        // 파도 높이 - Unity 기본값 0
        initialEnvironmentValues["WaveHeight"] = "-";
        initialSliderValues["WaveHeight"] = 0f;
        
        // 파도 속도 - Unity 기본값 0
        initialEnvironmentValues["WaveSpeed"] = "-";
        initialSliderValues["WaveSpeed"] = 0f;
        
        // 파도 진행 방향 NS - Unity 기본값 0
        initialSliderValues["WaveDirectionNS"] = 0f;
        
        // 파도 진행 방향 WE - Unity 기본값 0
        initialSliderValues["WaveDirectionWE"] = 0f;
        
        // 수중 가시 거리 - Unity 기본값 1
        initialEnvironmentValues["WaveClarity"] = "-";
        initialSliderValues["WaveClarity"] = 0f;
        
        // 부력 영향 정도 - Unity 기본값 1
        initialEnvironmentValues["BuoyancyStrength"] = "-";
        initialSliderValues["BuoyancyStrength"] = 0f;
        
        // 해수면 높이 - Unity 기본값 0
        initialEnvironmentValues["SeaLevel"] = "-";
        initialSliderValues["SeaLevel"] = 0f;
    }

    // 저장된 초기값으로 환경 설정을 복원하는 함수
    void RestoreInitialEnvironmentValues()
    {
        var env = GameManager.scenarioEdit.scinfo.scenarioEditInfo.environmentEdit.environmentvalue;
        
        // 텍스트 값들 복원 (Unity 기본값으로)
        if (env.LightingIntensity_T != null) 
            env.LightingIntensity_T.text = initialEnvironmentValues["LightingIntensity"];
        if (env.SunAngle_T != null) 
            env.SunAngle_T.text = initialEnvironmentValues["SunAngle"];
        if (env.Temperature_T != null) 
            env.Temperature_T.text = initialEnvironmentValues["Temperature"];
        if (env.RainIntensity_T != null) 
            env.RainIntensity_T.text = initialEnvironmentValues["RainIntensity"];
        if (env.Visibility_T != null) 
            env.Visibility_T.text = initialEnvironmentValues["Visibility"];
        if (env.WaveHeight_T != null) 
            env.WaveHeight_T.text = initialEnvironmentValues["WaveHeight"];
        if (env.WaveSpeed_T != null) 
            env.WaveSpeed_T.text = initialEnvironmentValues["WaveSpeed"];
        if (env.WaveClarity_T != null) 
            env.WaveClarity_T.text = initialEnvironmentValues["WaveClarity"];
        if (env.BuoyancyStrength_T != null) 
            env.BuoyancyStrength_T.text = initialEnvironmentValues["BuoyancyStrength"];
        if (env.SeaLevel_T != null) 
            env.SeaLevel_T.text = initialEnvironmentValues["SeaLevel"];

        // 슬라이더 값들 복원 (Unity 기본값으로)
        if (env.LightingIntensity_S != null) 
            env.LightingIntensity_S.value = initialSliderValues["LightingIntensity"];
        if (env.SunAngle_S != null) 
            env.SunAngle_S.value = initialSliderValues["SunAngle"];
        if (env.Temperature_S != null) 
            env.Temperature_S.value = initialSliderValues["Temperature"];
        if (env.RainIntensity_S != null) 
            env.RainIntensity_S.value = initialSliderValues["RainIntensity"];
        if (env.Visibility_S != null) 
            env.Visibility_S.value = initialSliderValues["Visibility"];
        if (env.WaveHeight_S != null) 
            env.WaveHeight_S.value = initialSliderValues["WaveHeight"];
        if (env.WaveSpeed_S != null) 
            env.WaveSpeed_S.value = initialSliderValues["WaveSpeed"];
        if (env.WaveClarity_S != null) 
            env.WaveClarity_S.value = initialSliderValues["WaveClarity"];
        if (env.BuoyancyStrength_S != null) 
            env.BuoyancyStrength_S.value = initialSliderValues["BuoyancyStrength"];
        if (env.SeaLevel_S != null) 
            env.SeaLevel_S.value = initialSliderValues["SeaLevel"];
        if (env.WaveDirectionNS != null) 
            env.WaveDirectionNS.value = initialSliderValues["WaveDirectionNS"];
        if (env.WaveDirectionWE != null) 
            env.WaveDirectionWE.value = initialSliderValues["WaveDirectionWE"];

        // 실제 환경에도 Unity 기본값 적용
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("lighting_intensity", "18000");
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("sun_angle", "123");
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("temperature", "6500");
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_height", initialEnvironmentValues["WaveHeight"]);
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_speed", initialEnvironmentValues["WaveSpeed"]);
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_direction_ns", initialSliderValues["WaveDirectionNS"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_direction_we", initialSliderValues["WaveDirectionWE"].ToString());
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("wave_clarity", initialEnvironmentValues["WaveClarity"]);
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("buoyancy_strength", initialEnvironmentValues["BuoyancyStrength"]);
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction?.Invoke("sea_level", initialEnvironmentValues["SeaLevel"]);
    }


}
