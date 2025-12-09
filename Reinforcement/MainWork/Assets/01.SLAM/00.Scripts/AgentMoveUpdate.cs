using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentMoveUpdate : MonoBehaviour
{
    public Rigidbody rigid;
    public WheelCollider wheel1, wheel2, wheel3, wheel4;
    public float drivespeed, steerspeed;
    public float rotationSpeed = 30.0f;

    [Header("Path Following Params")]
    public float linearSpeed = 1.0f;        // m/s
    public float angularSpeedDeg = 180f;    // deg/s
    public float waypointArriveDist = 0.10f;// m (도착 판정)
    public float slowDownRadius = 0.50f;    // m (가까워지면 감속)
    public float lookAheadTurnDeg = 5f;

    private Coroutine rlMoveCoroutine;

    private Vector3 prevPosition;
    private float prevRotationY;

    Vector3 anchorPosWorld;

    private void Awake() 
    {
        prevPosition = transform.position;
        prevRotationY = transform.eulerAngles.y;

        anchorPosWorld = transform.position;     

        UpdateDeltaPose();
    }
    void Start()
    {
        GameManager.s_agent.MoveUpdateAgent -= ApplyRLCommands;
        GameManager.s_agent.MoveUpdateAgent += ApplyRLCommands;
    }

    // RL 명령 리스트를 받아 순차적으로 실행하는 함수
    public void ApplyRLCommands(List<Vector3> waypoints)
    {
        StartCoroutine(ExecuteRLCommands(waypoints));
    }

    // y=0으로 납작하게
    Vector3 Flat(Vector3 v) => new Vector3(v.x, 0f, v.z);

    private IEnumerator ExecuteRLCommands(List<Vector3> waypointsFromPython)
    {
        var waypointsWorld = new List<Vector3>(waypointsFromPython.Count);
        float yKeep = transform.position.y; // 높이 유지
        for (int i = 0; i < waypointsFromPython.Count; i++)
        {
            Vector3 p = waypointsFromPython[i];

            float worldX = anchorPosWorld.x + p.z; // y => X
            float worldZ = anchorPosWorld.z + p.x; // x => Z

            waypointsWorld.Add(new Vector3(worldX, yKeep, worldZ));
        }

        int startIdx = 0;

        GameManager.s_agent.AgentState = GameManager.s_agent.RenewalState;
        for (int i = startIdx; i < waypointsWorld.Count; i++)
        {
            if (i == waypointsWorld.Count-1)
            {
                // 마지막 웨이포인트에 도달하면 상태를 PROCESS로 변경
                GameManager.s_agent.AgentState = GameManager.s_agent.ProcessState;
            }
            yield return StartCoroutine(ExecuteSingleCommand(waypointsWorld[i])); // 이동 + 통신/대기 수행
            
            // ExecuteSingleCommand가 Waypoint 도착으로 인해 종료되었을 때, 
            // 혹시 아직 처리되지 않은 통신이 있다면 여기서 한번 더 대기 (안전망)
            // while (GameManager.s_comm.s_comm_Coroutine != null)
            // {
            //     yield return null;
            // }
        }

        while (GameManager.s_comm.s_comm_Coroutine != null)
        {
            yield return null;
        }

        // ⭐ 최종 PROCESS 상태 전송 보장 (ExecuteSingleCommand가 마지막 통신을 놓쳤을 경우를 대비)
        if (GameManager.s_agent.AgentState == GameManager.s_agent.ProcessState)
        {
            UpdateDeltaPose();
            GameManager.s_agent.StartLidar?.Invoke();
            
            // PROCESS 데이터의 응답이 올 때까지 명시적으로 대기
            while (GameManager.s_comm.s_comm_Coroutine != null)
            {
                yield return null;
            }
        }
        
        yield break;
    } 

    private IEnumerator ExecuteSingleCommand(Vector3 waypointWorld)
    {
        Vector3 targetPosition = new Vector3(waypointWorld.x, transform.position.y, waypointWorld.z);

        bool isFirstLidarCall = true;
        
        while (true)
        {
            UpdateDeltaPose();
            if (isFirstLidarCall || GameManager.s_comm.s_comm_Coroutine == null)
            {
                isFirstLidarCall = false; 
                yield return new WaitForSeconds(0.03f); 
                GameManager.s_agent.StartLidar?.Invoke(); // LIDAR 스캔 -> RequestLoop 호출
            }
            else
            {
 
                yield return new WaitForSeconds(0.03f); 
            }

            
            if (Vector3.Distance(transform.position, targetPosition) < 0.05f)
            {
                yield break;
            }
            
            Vector3 posFlat = Flat(transform.position);
            Vector3 wpFlat  = Flat(waypointWorld);
            Vector3 to      = wpFlat - posFlat;
            float   dist    = to.magnitude;

            Vector3 dirFlat = to / dist; // normalized
            float targetYaw = Mathf.Atan2(dirFlat.x, dirFlat.z) * Mathf.Rad2Deg;
            Quaternion targetRot = Quaternion.Euler(0f, targetYaw, 0f);
            Quaternion newRot = Quaternion.RotateTowards(
                rigid.rotation, targetRot, angularSpeedDeg * Time.fixedDeltaTime
            );
            rigid.MoveRotation(newRot);
            // UpdateDeltaPose();
            // if (isFirstLidarCall || GameManager.s_comm.s_comm_Coroutine == null)
            // {
            //     isFirstLidarCall = false; 
            //     yield return new WaitForSeconds(0.05f); 
            //     GameManager.s_agent.StartLidar?.Invoke(); // LIDAR 스캔 -> RequestLoop 호출
            // }
            // else
            // {
 
            //     yield return new WaitForSeconds(0.05f); 
            // }
            
            transform.position = Vector3.MoveTowards(transform.position, targetPosition, drivespeed * Time.deltaTime);
            
            UpdateDeltaPose();
            if (isFirstLidarCall || GameManager.s_comm.s_comm_Coroutine == null)
            {
                isFirstLidarCall = false; 
                yield return new WaitForSeconds(0.03f); 
                GameManager.s_agent.StartLidar?.Invoke(); // LIDAR 스캔 -> RequestLoop 호출
            }
            else
            {
 
                yield return new WaitForSeconds(0.03f); 
            }
            
        }
    }


    private void UpdateDeltaPose()
    {
        Vector3 currPos = transform.position;
        float currYawDeg = transform.eulerAngles.y;

        // Δ위치
        Vector3 deltaPos = currPos - prevPosition;
        float deltaX_m = deltaPos.x;
        float deltaY_m = deltaPos.z;

        // Δ회전
        float deltaYawDeg = Mathf.DeltaAngle(prevRotationY, currYawDeg);
        float deltaTheta_rad = deltaYawDeg * Mathf.Deg2Rad;

        GameManager.s_agent.poseX_m += deltaY_m;
        GameManager.s_agent.poseY_m += deltaX_m;
        // GameManager.s_agent.poseX_m += deltaX_m;
        // GameManager.s_agent.poseY_m += deltaY_m;
        GameManager.s_agent.poseTheta_rad = deltaTheta_rad;

        prevPosition = currPos;
        prevRotationY = currYawDeg;
        
    }
}
