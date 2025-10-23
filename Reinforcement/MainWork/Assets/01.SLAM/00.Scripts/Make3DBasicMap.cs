using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using UnityEngine;
using System; 
using Newtonsoft.Json;

public class Make3DBasicMap : MonoBehaviour
{
    public float wallHeight = 15f;            
    public int occThreshold = 65;              
    public bool treat255AsUnknown = true;      // 255(전송 전 -1 치환)를 unknown으로 볼지 여부
    
    
    private OgmPacket _pkt;
    private byte[] _gridRaw;

    public GameObject GridMap_Position;

    public GameObject wallPrefab;
    public GameObject FreePrefab;

    public GameObject RobotPrefab;

    private string jsonfile_path = @"D:\Code\MainWork\MainWork\DigitalTwin\Exploration\exports\gridmap_jsonfile.json";

    void Start()
    {
        GameManager.s_map.StartMaking -= Load_2D_GridMap_File;
        GameManager.s_map.StartMaking += Load_2D_GridMap_File;
        
    }

    void Load_2D_GridMap_File()
    {
        
        try
        {
            if (!File.Exists(jsonfile_path))
            {
                Debug.LogError($"[OGM] JSON file not found:\n{jsonfile_path}");
                return;
            }

            // 1) 파일에서 JSON 텍스트 읽기
            string jsonText = File.ReadAllText(jsonfile_path, Encoding.UTF8);

            // 2) Newtonsoft로 파싱
            _pkt = JsonConvert.DeserializeObject<OgmPacket>(jsonText);
            if (_pkt == null)
            {
                Debug.LogError("[OGM] JsonConvert.DeserializeObject returned null.");
                return;
            }

            // 3) Base64 → GZIP 해제 → raw bytes (H*W)
            _gridRaw = DecodeGzipBase64(_pkt.data_gzip_b64);
            if (_gridRaw == null || _gridRaw.Length != _pkt.width * _pkt.height)
            {
                Debug.LogError($"[OGM] Byte length mismatch: got={_gridRaw?.Length}, expect={_pkt.width * _pkt.height}");
                return;
            }

            // // 4) 확인 로그
            // Debug.Log(
            //     $"[OGM] Loaded (Newtonsoft): {jsonfile_path}\n" +
            //     $"- robot: {_pkt.robot.x}, {_pkt.robot.y}, {_pkt.robot.theta} m\n" +
            //     $"- size: {_pkt.width}x{_pkt.height}, res={_pkt.resolution} m\n" +
            //     $"- origin: x={_pkt.origin.x}, y={_pkt.origin.y}, z={_pkt.origin.z}, yaw={_pkt.origin.yaw}\n" +
            //     $"- raw bytes: {_gridRaw.Length}, head: {SampleHead(_gridRaw, 8)}"
            // );

            // 5) 여기서 생성 로직 호출
            Make_3D_GridMap();
        }
        catch (Exception ex)
        {
            Debug.LogError($"[OGM] Exception: {ex}");
        }

    }

    // --------------------- Helper ----------------------
    static byte[] DecodeGzipBase64(string b64)
    {
        byte[] gz = Convert.FromBase64String(b64);
        using (var msIn = new MemoryStream(gz))
        using (var gzStream = new GZipStream(msIn, CompressionMode.Decompress))
        using (var msOut = new MemoryStream())
        {
            gzStream.CopyTo(msOut);
            return msOut.ToArray();
        }
    }

    static string SampleHead(byte[] arr, int n)
    {
        int m = Mathf.Min(n, arr.Length);
        var sb = new StringBuilder();
        for (int i = 0; i < m; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(arr[i]);
        }
        return sb.ToString();
    }
    // ---------------------------------------------------

    void Make_3D_GridMap()
    {
        if (_pkt == null || _gridRaw == null) { Debug.LogError("[OGM] No packet/raw data."); return; }

        int W = _pkt.width, H = _pkt.height;
        float res = (float)_pkt.resolution;

        Transform root = GridMap_Position ? GridMap_Position.transform : null;
        if (!root) { Debug.LogError("[OGM] GridMap_Position not assigned."); return; }

        // 그룹 준비(있으면 재사용)
        Transform wallsParent = root.Find("OGM_Walls") ?? new GameObject("OGM_Walls").transform;
        Transform freeParent  = root.Find("OGM_Free")  ?? new GameObject("OGM_Free").transform;
        Transform RobotParent  = root.Find("OGM_Robot")  ?? new GameObject("OGM_Robot").transform;
        
        wallsParent.SetParent(root, false);
        freeParent.SetParent(root, false);
        RobotParent.SetParent(root, false);

        float yBase = (float)_pkt.origin.z;

        GameObject wallPf = wallPrefab;
        GameObject freePf = FreePrefab;
        GameObject robotPf = RobotPrefab;

        int occRects = 0, freeRects = 0;

        

        // ------------------- 1) OCCUPIED(벽/장애물) 병합 스폰 -------------------
        if (wallPf != null)
        {
            bool[,] visitedOcc = new bool[H, W];

            for (int i = 0; i < H; i++)
            {
                for (int j = 0; j < W; j++)
                {
                    if (visitedOcc[i, j]) continue;
                    byte v = _gridRaw[i * W + j];
                    if (!IsOccupied(v)) continue;

                    // 가로 확장
                    int w = 0;
                    while (j + w < W && !visitedOcc[i, j + w] && IsOccupied(_gridRaw[i * W + (j + w)]))
                        w++;

                    // 세로 확장(폭 w 유지)
                    int h = 1; bool canGrow = true;
                    while (i + h < H && canGrow)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            if (visitedOcc[i + h, j + x] || !IsOccupied(_gridRaw[(i + h) * W + (j + x)]))
                            { canGrow = false; break; }
                        }
                        if (canGrow) h++;
                    }

                    // 중심(로컬)과 스케일
                    float cx = (j + (w * 0.5f)) * res;
                    float cz = (i + (h * 0.5f)) * res;

                    var go = Instantiate(wallPf, wallsParent);
                    go.name = $"occ_rect_{i}_{j}_{w}x{h}";

                    // ▶ 높이 고정 + 아래쪽으로 내려가도록 배치
                    Vector3 s = wallPf.transform.localScale;
                    s.x = w * res;
                    s.z = h * res;
                    s.y = wallHeight;                    // 높이를 고정
                    go.transform.localScale = s;

                    // yBase(=origin.z) 평면이 윗면이 되도록, 로컬 -Y 방향으로 절반만큼 내림
                    float yTop = yBase;                  // 윗면 기준 높이(필요시 0으로 두어도 됨)
                    go.transform.localPosition = new Vector3(cx, yTop - (s.y * 0.5f), cz);
                    go.transform.localRotation = Quaternion.identity;
                    go.SetActive(true);
                    occRects++;

                    // 방문 처리
                    for (int di = 0; di < h; di++)
                        for (int dj = 0; dj < w; dj++)
                            visitedOcc[i + di, j + dj] = true;
                }
            }
        }
        else
        {
            Debug.LogWarning("[OGM] wallPrefab is null. Occupied cells will not be spawned.");
        }

        // ------------------- 2) FREE(바닥) 병합 스폰 -------------------
        if (freePf != null)
        {
            bool[,] visitedFree = new bool[H, W];

            for (int i = 0; i < H; i++)
            {
                for (int j = 0; j < W; j++)
                {
                    if (visitedFree[i, j]) continue;
                    byte v = _gridRaw[i * W + j];
                    if (!IsFree(v)) continue;

                    // 가로 확장
                    int w = 0;
                    while (j + w < W && !visitedFree[i, j + w] && IsFree(_gridRaw[i * W + (j + w)]))
                        w++;

                    // 세로 확장(폭 w 유지)
                    int h = 1; bool canGrow = true;
                    while (i + h < H && canGrow)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            if (visitedFree[i + h, j + x] || !IsFree(_gridRaw[(i + h) * W + (j + x)]))
                            { canGrow = false; break; }
                        }
                        if (canGrow) h++;
                    }

                    // 중심(로컬)과 스케일
                    float cx = (j + (w * 0.5f)) * res;
                    float cz = (i + (h * 0.5f)) * res;

                    var go = Instantiate(freePf, freeParent);
                    go.name = $"free_rect_{i}_{j}_{w}x{h}";
                    Vector3 s = freePf.transform.localScale;
                    s.x = w * res;
                    s.z = h * res;
                    // go.transform.localScale = s*0.1f;
                    go.transform.localScale = s;

                    float y = yBase + s.y * 0.5f; // 바닥 타일 두께 만큼 올리기
                    go.transform.localPosition = new Vector3(cx, y, cz);
                    go.transform.localRotation = Quaternion.identity;
                    go.SetActive(true);
                    freeRects++;

                    // 방문 처리
                    for (int di = 0; di < h; di++)
                        for (int dj = 0; dj < w; dj++)
                            visitedFree[i + di, j + dj] = true;
                }
            }
        }
        else
        {
            Debug.LogWarning("[OGM] FreePrefab is null. Free tiles will not be spawned.");
        }

        if (robotPf != null)
        {
            // 1) 프리팹 인스턴스 생성 (부모: OGM_Robot)
            var rob = Instantiate(robotPf, RobotParent);

            // 2) 로컬 좌표 계산 (map_index 우선, 없으면 x,y 사용)
            float lx, lz;
            if (_pkt.robot.map_index != null)
            {
                lx = ((float)_pkt.robot.map_index.ix + 0.5f) * res; // 셀 중심
                lz = ((float)_pkt.robot.map_index.iy + 0.5f) * res;
            }
            else
            {
                lx = (float)(_pkt.robot.x - _pkt.origin.x); // world->local
                lz = (float)(_pkt.robot.y - _pkt.origin.y);
            }

            // 3) 바닥 높이 + 반높이 보정(프리팹 pivot이 중앙일 수 있음)
            rob.transform.localPosition = new Vector3(lx, -1f, lz);

            rob.transform.localRotation = Quaternion.Euler(0, 90, 180);

            // 4) 회전(Yaw만; 라디안)
            float theta = (float)_pkt.robot.theta;
            Vector3 dir = new Vector3(Mathf.Cos(theta), 0f, Mathf.Sin(theta));
            // if (dir.sqrMagnitude < 1e-6f) dir = Vector3.forward;
            // rob.transform.localRotation = Quaternion.LookRotation(dir, Vector3.up);

            rob.SetActive(true);

            // (선택) 디버그: 실제 배치 위치/회전 확인
            Debug.Log($"[OGM] Robot placed local=({rob.transform.localPosition}) world=({rob.transform.position}) theta={theta}");
        }

        
        root.transform.localRotation = Quaternion.Euler(0, 90, 180);
        var pos = root.transform.position;
        pos.z = (float)_pkt.origin.y;
        root.transform.position = pos;

        



        Debug.Log($"[OGM] Rect-merged: walls={occRects}, free={freeRects} (res={res})");
    }

    // free 판단: (255=unknown은 제외)
    bool IsFree(byte v)
    {
        if (treat255AsUnknown && v == 255) return false;
        return v < occThreshold;
    }


    bool IsOccupied(byte v)
    {
        if (treat255AsUnknown && v == 255) return false;
        return v >= occThreshold;
    }




}

[Serializable] 
public class MapIndex { 
    public int iy, ix; 
}

[Serializable]
public class RobotData  {
    public double x, y, z, theta;
    public MapIndex map_index; 
}

[Serializable]
public class Origin {
    public double x, y, z, yaw;
}

[Serializable]
public class OgmPacket {
    public RobotData robot;
    public int width;
    public int height;
    public double resolution;
    public Origin origin;
    public string data_gzip_b64;
}
