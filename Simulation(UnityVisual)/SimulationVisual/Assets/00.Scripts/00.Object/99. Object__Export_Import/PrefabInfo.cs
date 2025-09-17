using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class PrefabInfo
{
    [System.Serializable]
    public class ImportedObjectInfo
    {
        public string fileId;
        public string fileName;
        public string fileDesc;
        public string table;

        public ImportedObjectInfo(string fileId, string fileName, string fileDesc, string table)
        {
            this.fileId = fileId;
            this.fileName = fileName;
            this.fileDesc = fileDesc;
            this.table = table;
        }
    }

    [System.Serializable]
    public class ImportObjectUnityInfo
    {
        public string unityId;
        public string state;

        public ImportObjectUnityInfo(string unityId, string state)
        {
            this.unityId = unityId;
            this.state = state;
        }
    }

    [System.Serializable]
    public class WaypointInfo
    {
        public int index; // 웨이포인트 인덱스
        public GameObject pointObject; // 웨이포인트 오브젝트
        public GameObject pointButtonObject; // 웨이포인트 오브젝트
        public float object_x;
        public float object_z;
        public string endpoint; // 엔드포인트 정보(문자열)

        public WaypointInfo(int index, GameObject pointObject, GameObject pointButtonObject,float object_x, float object_z, string endpoint)
        {
            this.index = index;
            this.pointObject = pointObject;
            this.pointButtonObject = pointButtonObject;
            this.object_x = object_x;
            this.object_z = object_z;
            this.endpoint = endpoint;
        }
    }

    // Pre 오브젝트 정보를 저장하는 정적 딕셔너리
    private static Dictionary<GameObject, ImportedObjectInfo> importedObjectInfos = new Dictionary<GameObject, ImportedObjectInfo>();
    private static Dictionary<GameObject, ImportObjectUnityInfo> importedObjectUnityInfos = new Dictionary<GameObject, ImportObjectUnityInfo>();

    // waypointListnum을 키로, WaypointInfo를 값으로 하는 딕셔너리
    private static Dictionary<GameObject, List<WaypointInfo>> waypointsList = new Dictionary<GameObject, List<WaypointInfo>>();

    public static ImportedObjectInfo GetImportedObjectInfo(GameObject obj)
    {
        if (importedObjectInfos.TryGetValue(obj, out var info))
        {
            return info;
        }
        return null;
    }

    public static ImportObjectUnityInfo GetImportedObjectUnityInfo(GameObject obj)
    {
        if (importedObjectUnityInfos.TryGetValue(obj, out var unity_info))
        {
            return unity_info;
        }
        return null;
    }

    public static List<WaypointInfo> GetImportedObjectWayPointInfo(GameObject obj)
    {
        if (waypointsList.TryGetValue(obj, out var waypoint_info))
        {
            return waypoint_info;
        }
        return null;
    }

    // 딕셔너리에 정보를 추가하는 메서드
    public static void AddImportedObjectInfo(GameObject obj, ImportedObjectInfo info)
    {
        importedObjectInfos[obj] = info;
    }
    public static void AddImportedObjectUnityInfo(GameObject obj, ImportObjectUnityInfo unity_info)
    {
        importedObjectUnityInfos[obj] = unity_info;
    }

    // unityId로 오브젝트를 찾아 unity_info의 state를 업데이트
    public static bool UpdateImportedObjectUnityStateByUnityId(string unityId, string newState)
    {
        if (string.IsNullOrEmpty(unityId)) return false;
        var targetPair = importedObjectUnityInfos.FirstOrDefault(kvp => kvp.Value != null && kvp.Value.unityId == unityId);
        if (targetPair.Key == null) return false;
        targetPair.Value.state = newState;
        importedObjectUnityInfos[targetPair.Key] = targetPair.Value;
        return true;
    }
    
    public static void AddImportedObjectWayPointsInfo(GameObject obj, WaypointInfo waypoint_info)
    {
        if (!waypointsList.ContainsKey(obj))
        {
            waypointsList[obj] = new List<WaypointInfo>();
        }
        waypointsList[obj].Add(waypoint_info);
    }

    // 특정 테이블의 모든 파일 ID를 리스트로 반환하는 메서드
    public static List<string> GetFileIdsByTable(string table)
    {
        return importedObjectInfos.Values
            .Where(info => info.table == table)
            .Select(info => info.fileId)
            .ToList();
    }

    // 딕셔너리의 모든 정보를 확인하는 메서드 (디버깅용)
    public static void LogAllImportedObjects()
    {
        // Debug.Log($"총 import된 오브젝트 수: {importedObjectInfos.Count}");

        // 삭제할 fileId를 미리 모아둠
        List<string> fileIdsToRemove = new List<string>();

        foreach (var kvp in importedObjectInfos)
        {
            if (kvp.Value.table == "Agent")
            {
                // Debug.Log($"오브젝트: {kvp.Key.name}, 파일ID: {kvp.Value.fileId}, 파일명: {kvp.Value.fileName}, 테이블: {kvp.Value.table}");
                fileIdsToRemove.Add(kvp.Value.fileId);
            }
        }

        // 실제 삭제는 루프 이후에 진행
        foreach (var fileId in fileIdsToRemove)
        {
            RemoveImportedObjectInfoByFileId(fileId);
        }
    }

    // fileID로 연관된 모든 정보를 삭제하는 메서드
    public static void RemoveImportedObjectInfoByFileId(string fileId)
    {
        // fileId와 일치하는 오브젝트를 찾는다 (fileId는 info.fileId와 비교해야 함)
        var objToRemove = importedObjectInfos
            .FirstOrDefault(kvp => kvp.Value != null && kvp.Value.fileId == fileId)
            .Key;

        if (objToRemove != null)
        {
            // 딕셔너리에서 정보도 같이 삭제
            importedObjectInfos.Remove(objToRemove);
            importedObjectUnityInfos.Remove(objToRemove);
            // 해당 오브젝트의 웨이포인트 리스트가 있다면, 리스트 내의 웨이포인트 오브젝트들도 모두 삭제
            if (waypointsList.TryGetValue(objToRemove, out var waypointList))
            {
                foreach (var waypoint in waypointList)
                {
                    if (waypoint.pointObject != null)
                        UnityEngine.Object.Destroy(waypoint.pointObject);
                    if (waypoint.pointButtonObject != null)
                        UnityEngine.Object.Destroy(waypoint.pointButtonObject);
                }
            }
            waypointsList.Remove(objToRemove);

            UnityEngine.Object.Destroy(objToRemove);
        }
        else
        {
            Debug.LogWarning($"fileId: {fileId}에 해당하는 오브젝트를 찾을 수 없습니다.");
        }
    }

    // unityId에 해당하는 오브젝트를 찾아서 해당 오브젝트의 정보들을 삭제
    // fileId와 unityId가 모두 일치하는 오브젝트만 삭제
    public static void RemoveImportedObjectInfoByFileIdwithunityId(string unityId, string fileId)
    {
        // fileId와 unityId가 모두 일치하는 오브젝트를 찾는다
        var objToRemove = importedObjectInfos
            .FirstOrDefault(kvp =>
                kvp.Value != null &&
                kvp.Value.fileId == fileId &&
                importedObjectUnityInfos.TryGetValue(kvp.Key, out var unityInfo) &&
                unityInfo != null &&
                unityInfo.unityId == unityId
            ).Key;

        if (objToRemove != null)
        {
            // 딕셔너리에서 정보도 같이 삭제
            importedObjectInfos.Remove(objToRemove);
            importedObjectUnityInfos.Remove(objToRemove);
            // 해당 오브젝트의 웨이포인트 리스트가 있다면, 리스트 내의 웨이포인트 오브젝트들도 모두 삭제
            if (waypointsList.TryGetValue(objToRemove, out var waypointList))
            {
                foreach (var waypoint in waypointList)
                {
                    if (waypoint.pointObject != null)
                        UnityEngine.Object.Destroy(waypoint.pointObject);
                    if (waypoint.pointButtonObject != null)
                        UnityEngine.Object.Destroy(waypoint.pointButtonObject);
                }
            }
            waypointsList.Remove(objToRemove);

            UnityEngine.Object.Destroy(objToRemove);
        }
        else
        {
            Debug.LogWarning($"fileId: {fileId}와 unityId: {unityId}에 모두 해당하는 오브젝트를 찾을 수 없습니다.");
        }
    }

    public static (WaypointInfo, WaypointInfo) GetWaypointsByIndexAndEndpoint(GameObject obj, int index)
    {
        if (obj == null) return (null, null);

        if (waypointsList.TryGetValue(obj, out var waypointList))
        {
            WaypointInfo beforeWaypoint = null;
            WaypointInfo afterWaypoint = null;

            // before: index보다 1 작은 인덱스의 웨이포인트
            beforeWaypoint = waypointList.FirstOrDefault(w => w.index == index - 1);

            // after: index보다 1 큰 인덱스의 웨이포인트
            afterWaypoint = waypointList.FirstOrDefault(w => w.index == index + 1);

            // 리턴 조건에 따라 반환
            if (beforeWaypoint == null && afterWaypoint == null)
                return (null, null);
            else if (beforeWaypoint != null && afterWaypoint == null)
                return (beforeWaypoint, null);
            else if (beforeWaypoint == null && afterWaypoint != null)
                return (null, afterWaypoint);
            else // 둘 다 있으면
                return (beforeWaypoint, afterWaypoint);
        }
        return (null, null);
    }

    // 특정 오브젝트에서 인덱스 웨이포인트 제거 (구체/버튼 제거 및 리스트에서 삭제)
    public static void RemoveWaypoint(GameObject obj, int index)
    {
        if (obj == null) return;
        if (!waypointsList.TryGetValue(obj, out var waypointList)) return;

        var target = waypointList.FirstOrDefault(w => w.index == index);
        if (target == null) return;

        if (target.pointObject != null)
        {
            UnityEngine.Object.Destroy(target.pointObject);
        }
        if (target.pointButtonObject != null)
        {
            UnityEngine.Object.Destroy(target.pointButtonObject);
        }

        waypointList.Remove(target);
    }

    // 특정 웨이포인트의 endpoint와 버튼 UI(InputField)를 동기화하여 설정
    public static void SetWaypointEndpointAndInput(WaypointInfo waypoint, string endpoint)
    {
        if (waypoint == null) return;
        waypoint.endpoint = endpoint;

        if (waypoint.pointButtonObject != null)
        {
            Transform connectPanel = waypoint.pointButtonObject.transform.Find("Connect_Panel");
            if (connectPanel != null)
            {
                Transform connectInput = connectPanel.Find("ConnectInput");
                if (connectInput != null)
                {
                    var inputField = connectInput.GetComponent<UnityEngine.UI.InputField>();
                    if (inputField != null)
                    {
                        inputField.text = endpoint;
                    }
                }
            }
        }
    }

    // 특정 오브젝트에서 인덱스로 WaypointInfo 조회
    public static WaypointInfo GetWaypointByIndex(GameObject obj, int index)
    {
        if (obj == null) return null;
        if (waypointsList.TryGetValue(obj, out var waypointList))
        {
            return waypointList.FirstOrDefault(w => w.index == index);
        }
        return null;
    }

    // 전달된 오브젝트의 웨이포인트만 활성화하고, 다른 오브젝트의 웨이포인트는 비활성화
    public static void ToggleWaypointsExclusiveForObject(GameObject targetObject)
    {
        foreach (var pair in waypointsList)
        {
            bool isTarget = pair.Key == targetObject;
            var list = pair.Value;
            if (list == null) continue;
            foreach (var waypoint in list)
            {
                if (waypoint == null) continue;
                if (waypoint.pointObject != null)
                {
                    waypoint.pointObject.SetActive(isTarget);
                }
                if (waypoint.pointButtonObject != null)
                {
                    waypoint.pointButtonObject.SetActive(isTarget);
                }
            }
        }
    }


}


