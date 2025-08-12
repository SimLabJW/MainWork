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
        public string table;

        public ImportedObjectInfo(string fileId, string fileName, string table)
        {
            this.fileId = fileId;
            this.fileName = fileName;
            this.table = table;
        }
    }

    // Pre 오브젝트 정보를 저장하는 정적 딕셔너리
    private static Dictionary<GameObject, ImportedObjectInfo> importedObjectInfos = new Dictionary<GameObject, ImportedObjectInfo>();

    public static ImportedObjectInfo GetImportedObjectInfo(GameObject obj)
    {
        if (importedObjectInfos.TryGetValue(obj, out var info))
        {
            return info;
        }
        return null;
    }

    // 딕셔너리에 정보를 추가하는 메서드
    public static void AddImportedObjectInfo(GameObject obj, ImportedObjectInfo info)
    {
        importedObjectInfos[obj] = info;
    }

    // 딕셔너리에서 정보를 제거하는 메서드
    public static void RemoveImportedObjectInfo(GameObject obj)
    {
        if (importedObjectInfos.ContainsKey(obj))
        {
            importedObjectInfos.Remove(obj);
        }
    }

    // 딕셔너리의 모든 정보를 확인하는 메서드 (디버깅용)
    public static void LogAllImportedObjects()
    {
        Debug.Log($"총 import된 오브젝트 수: {importedObjectInfos.Count}");
        foreach (var kvp in importedObjectInfos)
        {
            Debug.Log($"오브젝트: {kvp.Key.name}, 파일ID: {kvp.Value.fileId}, 파일명: {kvp.Value.fileName}, 테이블: {kvp.Value.table}");
        }
    }


    // 특정 테이블의 모든 파일 ID를 리스트로 반환하는 메서드
    public static List<string> GetFileIdsByTable(string table)
    {
        return importedObjectInfos.Values
            .Where(info => info.table == table)
            .Select(info => info.fileId)
            .ToList();
    }

    [System.Serializable]
    public class ScenarioDefaultEnvInfo
    {
        public string WeatherType = "Sunny";
        public float LightingIntensity = 0.0f;
        public float SunAngle= 0.0f;
        public float Temperature= 0.0f;
        public float RainIntensity= 0.0f;
        public float Visibility= 0.0f;
        public float WaveHeight= 0.0f;
        public float WaveSpeed= 0.0f;
        public float WaveDirection_ns= 0.0f;
        public float WaveDirection_we= 0.0f;
        public float WaveClarity= 0.0f;
        public float BuoyancyStrength= 0.0f;
        public float SeaLevel= 0.0f;
    }

}