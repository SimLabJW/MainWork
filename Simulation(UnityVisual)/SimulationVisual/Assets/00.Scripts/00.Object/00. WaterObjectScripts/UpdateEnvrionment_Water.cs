using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

public class UpdateEnvrionment_Water : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction -= UpdateWaterObject;
        GameManager.scenarioEdit.UpdateWaterEnvrionmentAction += UpdateWaterObject;
    }

    void UpdateWaterObject(string type, string value)
    {
        switch(type)
        {
            case "wave_height": 
                if (float.TryParse(value, out float waveHeightValue))
                {
                    var mgr = GameManager.scenarioEdit;
                    var loaded = mgr != null ? mgr.LoadedWaterObject : null;
                    var ws = loaded != null ? loaded.GetComponent<WaterSurface>() : null;
                    if (ws != null)
                    {
                        ws.largeWindSpeed = waveHeightValue;
                    }
                }
                break;
            case "wave_speed":
                if (float.TryParse(value, out float waveSpeedValue))
                {
                    var mgr = GameManager.scenarioEdit;
                    var loaded = mgr != null ? mgr.LoadedWaterObject : null;
                    var ws = loaded != null ? loaded.GetComponent<WaterSurface>() : null;
                    if (ws != null)
                    {
                        ws.largeChaos = waveSpeedValue;
                    }
                }
                break;
            case "wave_direction_ns":
                // Debug.Log("WaveDirectionNS 값: " + value);
                break;
            case "wave_direction_we":
                // Debug.Log("WaveDirectionWE 값: " + value);
                break;
            case "wave_clarity":
                if (float.TryParse(value, out float clarityValue))
                {
                    var mgr = GameManager.scenarioEdit;
                    var loaded = mgr != null ? mgr.LoadedWaterObject : null;
                    var ws = loaded != null ? loaded.GetComponent<WaterSurface>() : null;

                    if (ws != null)
                    {
                        Debug.Log("1");
                        ws.underWaterAmbientProbeContribution = Mathf.Clamp01(clarityValue);
                    }
                }
                break;
            case "buoyancy_strength":
                if (float.TryParse(value, out float buoyancyStrength))
                    GameManager.scenarioEdit.BuoyancyStrength = buoyancyStrength;
                break;
            case "sea_level":
                if (float.TryParse(value, out float sea_level) && GameManager.scenarioEdit.LoadedWaterObject)
                    GameManager.scenarioEdit.LoadedWaterObject.transform.localScale = new Vector3(
                        GameManager.scenarioEdit.LoadedWaterObject.transform.localScale.x,
                        sea_level,
                        GameManager.scenarioEdit.LoadedWaterObject.transform.localScale.z
                    );
                break;

            case "lighting_intensity":
                // Directional Light의 Intensity 값을 조절
                if (float.TryParse(value, out float intensityValue))
                {
                    Light dirLight = GameObject.FindObjectOfType<Light>();
                    if (dirLight != null && dirLight.type == LightType.Directional)
                    {
                        dirLight.intensity = intensityValue;
                    }
                }
                break;
            case "sun_angle":
                // Directional Light의 각도(회전)를 조절 (Y축 기준)
                if (float.TryParse(value, out float sunAngleValue))
                {
                    Light dirLight = GameObject.FindObjectOfType<Light>();
                    if (dirLight != null && dirLight.type == LightType.Directional)
                    {
                        Vector3 euler = dirLight.transform.eulerAngles;
                        dirLight.transform.eulerAngles = new Vector3(sunAngleValue, euler.y, euler.z);
                    }
                }
                break;
            case "temperature":
                // Directional Light의 색온도(Temperature)를 조절
                if (float.TryParse(value, out float tempValue))
                {
                    Light dirLight = GameObject.FindObjectOfType<Light>();
                    if (dirLight != null && dirLight.type == LightType.Directional)
                    {
                        dirLight.colorTemperature = tempValue;
                        dirLight.useColorTemperature = true;
                    }
                }
                break;
            default:
                Debug.Log("알 수 없는 타입: " + type + ", 값: " + value);
                break;
        }
    }
}
