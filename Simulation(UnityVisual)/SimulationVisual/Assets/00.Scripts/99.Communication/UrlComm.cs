using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class UrlComm : MonoBehaviour
{
    void Start()
    {
        GameManager.communication.OnCommData -= CommunicationUrl; 
        GameManager.communication.OnCommData += CommunicationUrl; 

        GameManager.communication.OnCommSaveData -= CommunicationSaveUrl; 
        GameManager.communication.OnCommSaveData += CommunicationSaveUrl; 

        GameManager.communication.ScenarioCommAction -=CommunicationImportScenarioUrl;
        GameManager.communication.ScenarioCommAction +=CommunicationImportScenarioUrl;

        //insert
        GameManager.communication.ScenarioAgentInsertAction -= CommunicationScenarioInsertUrl;
        GameManager.communication.ScenarioAgentInsertAction += CommunicationScenarioInsertUrl;

        //delete
        GameManager.communication.ScenarioDeleteAction -= CommunicationScenarioDeleteUrl;
        GameManager.communication.ScenarioDeleteAction += CommunicationScenarioDeleteUrl;

        //find
        GameManager.communication.ScenarioInfoFindAction -= CommunicationScenarioFindIDUrl;
        GameManager.communication.ScenarioInfoFindAction += CommunicationScenarioFindIDUrl;

        //update
        GameManager.communication.ScenarioUpdateAction -= CommunicationScenarioUpdateUrl;
        GameManager.communication.ScenarioUpdateAction += CommunicationScenarioUpdateUrl;

    }

    // 데이터를 서버로 전송
    public void CommunicationUrl(string path, string table, List<string> columns, Dictionary<string, object> filters, string commmethod)
    {
        StartCoroutine(CommDataCoroutine(path, table, columns, filters, commmethod));
    }

    private IEnumerator CommDataCoroutine(string path, string table, List<string> columns, Dictionary<string, object> filters, string commmethod)
    {
    
        // 요청할 데이터를 JSON 형태로 구성
        var requestData = new
        {
            table = table,
            columns = columns,
            filters = filters
        };

        string jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(requestData);

        UnityWebRequest request = null;

        if (commmethod.ToUpper() == "POST")
        {
            // POST 방식: body에 json 데이터 실어서 전송
            request = new UnityWebRequest(path + "read/", "POST");
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("데이터 전송 실패: " + request.error);
            }
            else
            {
                string responseText = request.downloadHandler.text;
                var jArray = Newtonsoft.Json.Linq.JArray.Parse(responseText);

                // 첫 번째 요소는 문자열이므로, 다시 파싱 필요
                string firstElement = jArray[0]?.ToString();
                var innerObj = Newtonsoft.Json.Linq.JObject.Parse(firstElement);

                string resultString = innerObj["result"]?.ToString();

                GameManager.communication.Url_result = resultString;
            }
        }
        else if (commmethod.ToUpper() == "GET")
        {
            string idValue = "";
            if (filters != null && filters.ContainsKey("id"))
            {
                idValue = filters["id"]?.ToString();
            }
            else
            {
                Debug.LogError("GET 요청에 id 값이 필요합니다.");
                yield break;
            }

            // 쿼리 스트링 구성
            string query = $"?id={idValue}&table={table}";
            string fullUrl = path + "data-by-glb/" + query;

            request = UnityWebRequest.Get(fullUrl);
            request.SetRequestHeader("accept", "application/json");
            request.downloadHandler = new DownloadHandlerBuffer();

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("데이터 전송 실패: " + request.error);
            }
            else
            {
                // JSON 파싱 오류 수정: 배열 구조를 먼저 파싱한 뒤 내부 문자열을 다시 파싱
                string responseText = request.downloadHandler.text;

                // 1. 최상위는 JArray임
                var jArray = Newtonsoft.Json.Linq.JArray.Parse(responseText);

                // 2. 첫 번째 요소는 문자열(내부 JSON)임
                string firstElement = jArray[0]?.ToString();

                // 3. 내부 JSON 문자열을 JObject로 파싱
                var innerObj = Newtonsoft.Json.Linq.JObject.Parse(firstElement);

                // 4. "result"는 파이썬 dict의 str()처럼 싱글 쿼트(')와 b''(바이트) 표기가 섞여있어서 바로 JSON 파싱이 불가함
                string resultStringRaw = innerObj["result"]?.ToString();

                // 5. 싱글 쿼트(')를 더블 쿼트(")로 변환하고, b''(바이트) 표기를 제거
                // b'...' 형태를 "..."로 변환
                string cleaned = resultStringRaw
                    .Replace("b'", "\"")
                    .Replace("',", "\",")
                    .Replace("'}", "\"}")
                    .Replace("':", "\":")
                    .Replace("', '", "\", \"")
                    .Replace("'", "\"");

                // 6. 정상적인 JSON 배열로 파싱
                var resultArray = Newtonsoft.Json.Linq.JArray.Parse(cleaned);

                // 7. 첫 번째 객체에서 glb_data 추출
                var firstObj = resultArray.First as Newtonsoft.Json.Linq.JObject;
                string glbString = firstObj?["glb_data"]?.ToString();

                GameManager.communication.Url_result = glbString;
            }
        }  
    }

    // Scenario 저장
    public void CommunicationSaveUrl(string path, string table, string[] data)
    {
        StartCoroutine(CommSaveDataCoroutine(path, table, data));
    }

    private IEnumerator CommSaveDataCoroutine(string path, string table, string[] data)
    {
        // 환경 관련 모든 데이터를 하나의 env 오브젝트에 넣어서 전송
        var default_env = new
        {
            buoyancy_strength = 1,
            lighting_intensity = 18000,
            rain_intensity = 0,
            sea_level = 50,
            sun_angle = 123,
            temperature = 6500,
            visibility = 0,
            wave_clarity = 1, // water appearance ambient Probe color 
            wave_direction_ns = 0,
            wave_direction_we = 0,
            wave_height = 0, // water simulation distant wind speed
            wave_speed = 0, // water simulation chaos 
            weather_type = "Sunny"
        };

        List<int> scenarioAgentList = new List<int>();
        if (!string.IsNullOrEmpty(data[3]))
        {
            foreach (var s in data[3].Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
            {
                if (int.TryParse(s, out int id))
                    scenarioAgentList.Add(id);
            }
        }

        List<int> scenarioTerrainList = new List<int>();
        if (!string.IsNullOrEmpty(data[2]))
        {
            foreach (var s in data[2].Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
            {
                if (int.TryParse(s, out int id))
                    scenarioTerrainList.Add(id);
            }
        }

        // scenario_env는 현재 빈 배열로 처리
        List<int> scenarioEnvList = new List<int>();

        // 요청할 데이터를 JSON 형태로 구성
        var requestData = new
        {
            glb_name = data[0],
            glb_desc = data[1],
            glb_data = data[4],
            env = default_env, // env 오브젝트 전체를 넣음
            scenario_name = data[0],
            scenario_desc = data[1],
            scenario_agent = scenarioAgentList,
            scenario_terrain = scenarioTerrainList,
            scenario_env = scenarioEnvList
        };

        string jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(requestData);
        Debug.Log($"jsonData : {jsonData}");

        UnityWebRequest request = null;

        // POST 방식: body에 json 데이터 실어서 전송
        request = new UnityWebRequest(path + "insert-scenario/", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return null;
        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("데이터 전송 실패: " + request.error);
        }
        else
        {
            string responseText = request.downloadHandler.text;
            Debug.Log($"responeText : {responseText}");
        }
        
    }

    public void CommunicationImportScenarioUrl(string path, string scenarioId)
    {
        StartCoroutine(CommScenarioDataCoroutine(path, scenarioId));
    }
    private IEnumerator CommScenarioDataCoroutine(string path, string scenarioId)
    {
        UnityWebRequest request = null;
        // 쿼리 스트링 구성
        string query = $"?id={scenarioId}";
        string fullUrl = path + "scenario-by-glb/" + query;

        request = UnityWebRequest.Get(fullUrl);
        request.SetRequestHeader("accept", "application/json");
        request.downloadHandler = new DownloadHandlerBuffer();

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("데이터 전송 실패: " + request.error);
        }
        else
        {
            // JSON 파싱 오류 수정: 배열 구조를 먼저 파싱한 뒤 내부 문자열을 다시 파싱
            string responseText = request.downloadHandler.text;
            Debug.Log($"Get original : {responseText}");

            // 1. 최상위는 JArray임
            var jArray = Newtonsoft.Json.Linq.JArray.Parse(responseText);

            // 2. 첫 번째 요소는 문자열(내부 JSON)임
            string firstElement = jArray[0]?.ToString();

            // 3. 내부 JSON 문자열을 JObject로 파싱
            var innerObj = Newtonsoft.Json.Linq.JObject.Parse(firstElement);

            // 4. "result"는 파이썬 dict의 str()처럼 싱글 쿼트(')와 b''(바이트) 표기가 섞여있어서 바로 JSON 파싱이 불가함
            string resultStringRaw = innerObj["result"]?.ToString();

            // 5. 싱글 쿼트(')를 더블 쿼트(")로 변환하고, b''(바이트) 표기를 제거
            // b'...' 형태를 "..."로 변환
            string cleaned = resultStringRaw
                .Replace("b'", "\"")
                .Replace("',", "\",")
                .Replace("'}", "\"}")
                .Replace("':", "\":")
                .Replace("', '", "\", \"")
                .Replace("'", "\"");

            // 6. 정상적인 JSON 배열로 파싱
            var resultArray = Newtonsoft.Json.Linq.JArray.Parse(cleaned);

            // scenarioInfo 안에 agent 정보 리스트를 포함하도록 구조 변경
            Dictionary<string, object> scenarioInfo = null;
            List<Dictionary<string, object>> agentList = new List<Dictionary<string, object>>();

            for (int i = 0; i < resultArray.Count; i++)
            {
                var obj = resultArray[i] as Newtonsoft.Json.Linq.JObject;
                if (obj != null)
                {
                    // scenarioInfo는 첫 번째 루프에서만 세팅 (agentList 포함)
                    if (scenarioInfo == null)
                    {
                        scenarioInfo = new Dictionary<string, object>
                        {
                            { "scenario_id", obj["scenario_id"] },
                            { "scenario_name", obj["scenario_name"] },
                            { "scenario_desc", obj["scenario_desc"] },
                            { "terrian_id", obj["terrian_id"] },
                            { "terrian_name", obj["terrian_name"] },
                            { "terrian_desc", obj["terrian_desc"] },
                            { "terrian_glb_id", obj["terrian_glb_id"] },
                            { "env_id", obj["env_id"] },
                            { "env_name", obj["env_name"] },
                            { "weather_type", obj["weather_type"] },
                            { "lighting_intensity", obj["lighting_intensity"] },
                            { "sun_angle", obj["sun_angle"] },
                            { "temperature", obj["temperature"] },
                            { "rain_intensity", obj["rain_intensity"] },
                            { "visibility", obj["visibility"] },
                            { "wave_height", obj["wave_height"] },
                            { "wave_speed", obj["wave_speed"] },
                            { "wave_direction_ns", obj["wave_direction_ns"] },
                            { "wave_direction_we", obj["wave_direction_we"] },
                            { "wave_clarity", obj["wave_clarity"] },
                            { "buoyancy_strength", obj["buoyancy_strength"] },
                            { "sea_level", obj["sea_level"] },
                            { "agents", agentList } // agent 정보 리스트를 scenarioInfo에 포함
                        };
                    }

                    // agent 관련 정보는 매 루프마다 리스트에 추가
                    var agentInfo = new Dictionary<string, object>
                    {
                        { "agent_id", obj["agent_id"] },
                        { "agent_name", obj["agent_name"] },
                        { "agent_desc", obj["agent_desc"] }
                        // { "agent_glb_id", obj["agent_glb_id"] }
                    };
                    agentList.Add(agentInfo);
                }
            }

            GameManager.scenarioEdit.scenario_info = scenarioInfo;
            
        }
        
    }
    // Scenario Info Insert 
    public void CommunicationScenarioInsertUrl(string path, string table, Dictionary<string, object> columns)
    {
        StartCoroutine(CommScenarioInsertCoroutine(path, table, columns));
    }

    // Scenario Info Insert  
    private IEnumerator CommScenarioInsertCoroutine(string path, string table, Dictionary<string, object> columns)
    {
        var requestData = new
        {
            table = table,
            data = columns
        };

        string jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(requestData);

        UnityWebRequest request = null;

        request = new UnityWebRequest(path + "insert/", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("데이터 전송 실패: " + request.error);
        }
        else
        {
            string responseText = request.downloadHandler.text;
            Debug.Log($"Delete Result : {responseText}");
        }
        
    }

    // Scenario Info Delete 
    public void CommunicationScenarioDeleteUrl(string path, string table, Dictionary<string, object> filters)
    {
        StartCoroutine(CommScenarioDeleteCoroutine(path, table, filters));
    }

    // Scenario Info Delete 
    private IEnumerator CommScenarioDeleteCoroutine(string path, string table, Dictionary<string, object> filters)
    {
        Debug.Log($"path :{path} // table : {table}");

        // 요청할 데이터를 JSON 형태로 구성
        var requestData = new
        {
            table = table,
            filters = filters
        };

        string jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(requestData);
        Debug.Log($"jsonData : {jsonData}");

        UnityWebRequest request = null;

        
        // POST 방식: body에 json 데이터 실어서 전송
        request = new UnityWebRequest(path + "delete/", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("데이터 전송 실패: " + request.error);
        }
        else
        {
            string responseText = request.downloadHandler.text;
            Debug.Log($"Delete Result : {responseText}");
        }
        
    }

    // Scenario Update
    public void CommunicationScenarioUpdateUrl(string path, string table, Dictionary<string, object> columns, Dictionary<string, object> filters)
    {
        StartCoroutine(CommScenarioUpdateCoroutine(path, table, columns, filters));
    }
    // Scenario Update
    private IEnumerator CommScenarioUpdateCoroutine(string path, string table, Dictionary<string, object> columns, Dictionary<string, object> filters)
    {
        var requestData = new
        {
            table = table,
            data = columns,  
            filters = filters
        };

        string jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(requestData);
 
        UnityWebRequest request = new UnityWebRequest(path + "update/", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return null;
        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"데이터 전송 실패: {request.error} (HTTP 상태코드: {request.responseCode})");
        }
        else
        {
            string responseText = request.downloadHandler.text;
            // Debug.Log($"responeText : {responseText}");
        }
        
    }

    // ScenarioGLB Find
    public void CommunicationScenarioFindIDUrl(string path, string table, List<string> columns, Dictionary<string, object> filters)
    {
        StartCoroutine(CommScenarioFindIDCoroutine(path, table, columns, filters));
    }

    private IEnumerator CommScenarioFindIDCoroutine(string path, string table, List<string> columns, Dictionary<string, object> filters)
    {

        var requestData = new
        {
            table = table,
            columns = columns,
            filters = filters
        };

        string jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(requestData);
        Debug.Log($"jsonData : {jsonData}");

        UnityWebRequest request = null;

        // POST 방식: body에 json 데이터 실어서 전송
        request = new UnityWebRequest(path + "read/", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("데이터 전송 실패: " + request.error);
        }
        else
        {
            string responseText = request.downloadHandler.text;
            var jArray = Newtonsoft.Json.Linq.JArray.Parse(responseText);
            string firstElement = jArray[0]?.ToString();
            var firstObj = Newtonsoft.Json.Linq.JObject.Parse(firstElement);

            string resultStr = firstObj["result"]?.ToString();
            // 작은따옴표를 큰따옴표로 변환 (JSON 파싱을 위해)
            resultStr = resultStr.Replace('\'', '\"');
            var resultJArray = Newtonsoft.Json.Linq.JArray.Parse(resultStr);

            if (table == "Scenario")
            {
                if (resultJArray.Count > 0)
                {
                    Debug.Log($"Scenario debug : {resultStr}");
                    var glbId = resultJArray[0]["glb_id"];
                    var glbIdDict = new Dictionary<string, object> { { "id", glbId } };
                    GameManager.scenarioEdit.Scenario_GLBUpdateAction?.Invoke("GLB", glbIdDict);
                }
            }
            else if(table == "Scenario_Agent")
            {
                Debug.Log($"Scenario_Agent debug : {resultStr}");
                
                // resultJArray를 Dictionary 리스트로 변환
                var scenarioAgentList = new List<Dictionary<string, object>>();
                foreach (var item in resultJArray)
                {
                    var dict = new Dictionary<string, object>();
                    foreach (var prop in item)
                    {
                        dict[((Newtonsoft.Json.Linq.JProperty)prop).Name] = ((Newtonsoft.Json.Linq.JProperty)prop).Value;
                    }
                    scenarioAgentList.Add(dict);
                }
                
                // currentAgentList와 비교하여 계산
                var currentAgentList = GameManager.scenarioEdit.scenario_agentListDict;
                
                // 1. scenarioAgentList - currentAgentList (중복 고려, 남는 것만)
                // scenarioAgentList에서 currentAgentList에 있는 agent_id들을 제거 (id가 큰 것부터)
                var onlyInScenario = MultisetDifferenceByKey(scenarioAgentList, currentAgentList, "agent_id");
                
                // 2. currentAgentList - scenarioAgentList (중복 고려, 남는 것만)
                // currentAgentList에서 scenarioAgentList에 있는 agent_id들을 제거
                var onlyInCurrent = MultisetDifferenceByKey(currentAgentList, scenarioAgentList, "agent_id");
                
                // 결과를 GameManager에 저장
                // onlyInInitialAgentIds: {id, agent_id} 형태로 저장 (scenarioAgentList에서 남은 것들)
                GameManager.scenarioEdit.onlyInInitialAgentIds = new List<string>();
                foreach (var a in onlyInScenario)
                {
                    if (a.ContainsKey("id") && a.ContainsKey("agent_id"))
                    {
                        var id = a["id"] != null ? a["id"].ToString() : "null";
                        var agentId = a["agent_id"] != null ? a["agent_id"].ToString() : "null";
                        GameManager.scenarioEdit.onlyInInitialAgentIds.Add($"{{{id}, {agentId}}}");
                    }
                    else
                        GameManager.scenarioEdit.onlyInInitialAgentIds.Add("null");
                }
                
                // onlyInCurrentAgentIds: agent_id만 저장 (currentAgentList에서 남은 것들)
                GameManager.scenarioEdit.onlyInCurrentAgentIds = new List<string>();
                foreach (var a in onlyInCurrent)
                {
                    if (a.ContainsKey("agent_id"))
                        GameManager.scenarioEdit.onlyInCurrentAgentIds.Add(a["agent_id"] != null ? a["agent_id"].ToString() : "null");
                    else
                        GameManager.scenarioEdit.onlyInCurrentAgentIds.Add("null");
                }
                
                // Delete ScenarioAgent Relation
                Debug.Log("scenarioAgentList - currentAgentList 결과: " + string.Join(", ", GameManager.scenarioEdit.onlyInInitialAgentIds));
                foreach (var a in onlyInScenario)
                {
                    if (a.ContainsKey("id") && a.ContainsKey("agent_id"))
                    {
                        var filter = new Dictionary<string, object>
                        {
                            { "id", a["id"] },
                            { "scenario_id", GameManager.scenarioEdit.ScenarioId },
                            { "agent_id", a["agent_id"] }
                        };
                        GameManager.communication.ScenarioDelete("Scenario_Agent", filter);
                    }
                }
                
                // Insert ScenarioAgent Relation
                Debug.Log("currentAgentList - scenarioAgentList 결과: " + string.Join(", ", GameManager.scenarioEdit.onlyInCurrentAgentIds));
                foreach (var agentId in GameManager.scenarioEdit.onlyInCurrentAgentIds)
                {
                    if (agentId != "null")
                    {
                        var insertData = new Dictionary<string, object>
                        {
                            { "scenario_id", GameManager.scenarioEdit.ScenarioId },
                            { "agent_id", agentId }
                        };
                        GameManager.communication.ScenarioAgentInsert("Scenario_Agent", insertData);
                    }
                }
            }
        }
        
    }

    // 멀티셋 차집합(중복 고려) 유틸 - key 기준으로 동일 항목을 카운트 기반으로 제거
    // left: currentList (agent_id만 있음), right: scenarioAgentList (id, agent_id 모두 있음)
    private static List<Dictionary<string, object>> MultisetDifferenceByKey(
        List<Dictionary<string, object>> left,
        List<Dictionary<string, object>> right,
        string key)
    {
        // right(scenarioAgentList)를 agent_id별로 그룹화하고, 각 그룹 내에서 id가 작은 순으로 정렬
        var rightGroups = new Dictionary<string, List<Dictionary<string, object>>>();
        foreach (var r in right)
        {
            if (r == null || !r.ContainsKey(key)) continue;
            var agentId = r[key]?.ToString()?.Trim();
            if (string.IsNullOrEmpty(agentId)) continue;
            
            if (!rightGroups.ContainsKey(agentId))
                rightGroups[agentId] = new List<Dictionary<string, object>>();
            
            rightGroups[agentId].Add(r);
        }
        
        // 각 그룹을 id 순으로 정렬 (id가 큰 것부터)
        foreach (var group in rightGroups.Values)
        {
            group.Sort((a, b) => {
                if (!a.ContainsKey("id") || !b.ContainsKey("id")) return 0;
                if (int.TryParse(a["id"]?.ToString(), out int idA) && int.TryParse(b["id"]?.ToString(), out int idB))
                    return idB.CompareTo(idA); // id가 큰 것부터 정렬
                return 0;
            });
        }

        // left(currentList)에서 rightCounts를 소모하면서 남는 것만 결과에 추가
        var result = new List<Dictionary<string, object>>();
        foreach (var l in left)
        {
            if (l == null || !l.ContainsKey(key))
            {
                result.Add(l);
                continue;
            }
            
            var agentId = l[key]?.ToString()?.Trim();
            if (string.IsNullOrEmpty(agentId))
            {
                result.Add(l);
                continue;
            }

            // 해당 agent_id 그룹이 있고, 아직 제거할 항목이 남아있으면 제거
            if (rightGroups.TryGetValue(agentId, out var group) && group.Count > 0)
            {
                group.RemoveAt(0); // id가 가장 작은 것부터 제거
            }
            else
            {
                result.Add(l); // 상쇄할 것이 없으면 남김
            }
        }

        return result;
    }

}
