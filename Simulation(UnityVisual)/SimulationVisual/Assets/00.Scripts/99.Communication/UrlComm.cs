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
    }

    // 데이터를 서버로 전송
    public void CommunicationUrl(string path, string table, List<string> columns, Dictionary<string, object> filters, string commmethod)
    {
        StartCoroutine(CommDataCoroutine(path, table, columns, filters, commmethod));
    }

    private IEnumerator CommDataCoroutine(string path, string table, List<string> columns, Dictionary<string, object> filters, string commmethod)
    {
        Debug.Log($"path :{path} // table : {table} // columns : {string.Join(", ", columns)}");

        // 요청할 데이터를 JSON 형태로 구성
        var requestData = new
        {
            table = table,
            columns = columns,
            filters = filters
        };

        string jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(requestData);
        Debug.Log($"jsonData : {jsonData}");

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

                // 7. 첫 번째 객체에서 glb_data 추출
                var firstObj = resultArray.First as Newtonsoft.Json.Linq.JObject;
                string glbString = firstObj?["glb_data"]?.ToString();

                GameManager.communication.Url_result = glbString;
            }
        }  
    }

    // 데이터를 서버로 전송
    public void CommunicationSaveUrl(string path, string table, string[] data)
    {
        StartCoroutine(CommSaveDataCoroutine(path, table, data));
    }

    private IEnumerator CommSaveDataCoroutine(string path, string table, string[] data)
    {
        // 요청할 데이터를 JSON 형태로 구성
        var requestData = new
        {
            table = table,
            data = data
        };

        string jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(requestData);
        Debug.Log($"jsonData : {jsonData}");

        // UnityWebRequest request = null;

        // // POST 방식: body에 json 데이터 실어서 전송
        // request = new UnityWebRequest(path + "insert/", "POST");
        // byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
        // request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        // request.downloadHandler = new DownloadHandlerBuffer();
        // request.SetRequestHeader("Content-Type", "application/json");

        yield return null;
        // yield return request.SendWebRequest();

        // if (request.result != UnityWebRequest.Result.Success)
        // {
        //     Debug.LogError("데이터 전송 실패: " + request.error);
        // }
        // else
        // {
        //     string responseText = request.downloadHandler.text;
        //     Debug.Log("responeText");
        // }
        
    }

}
