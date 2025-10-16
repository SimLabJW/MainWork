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
        if (rlMoveCoroutine != null)
            StopCoroutine(rlMoveCoroutine);

        rlMoveCoroutine = StartCoroutine(ExecuteRLCommands(waypoints));
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
            if (i == waypointsWorld.Count -1)
            {
                GameManager.s_agent.AgentState = GameManager.s_agent.ProcessState;
            }
            yield return StartCoroutine(ExecuteSingleCommand(waypointsWorld[i]));
        }
        

        StopMovement();
        UpdateDeltaPose();                       // 완료 후 한 번 더
        // GameManager.s_agent.StartLidar?.Invoke();
        rlMoveCoroutine = null;
    } 

    private IEnumerator ExecuteSingleCommand(Vector3 waypointWorld)
    {
        // var wait = new WaitForFixedUpdate();

        const float posEps = 1e-4f; // 수치적 오차 허용(거의 0)

        while (true)
        {
            // 수평 좌표만 사용
            Vector3 posFlat = Flat(transform.position);
            Vector3 wpFlat  = Flat(waypointWorld);
            Vector3 to      = wpFlat - posFlat;
            float   dist    = to.magnitude;

            // 1) 위치가 사실상 같으면(수치 오차 수준) 정확히 스냅하고 종료
            if (dist <= posEps)
            {
                // 최종 스냅(정확히 목표 좌표로)
                rigid.MovePosition(new Vector3(wpFlat.x, rigid.position.y, wpFlat.z));
                yield break;
            }

            // 2) 목표 방향으로 회전(yaw만)
            Vector3 dirFlat = to / dist; // normalized
            float targetYaw = Mathf.Atan2(dirFlat.x, dirFlat.z) * Mathf.Rad2Deg;
            Quaternion targetRot = Quaternion.Euler(0f, targetYaw, 0f);
            Quaternion newRot = Quaternion.RotateTowards(
                rigid.rotation, targetRot, angularSpeedDeg * Time.fixedDeltaTime
            );
            rigid.MoveRotation(newRot);


            float speed = linearSpeed;

            float headingErr = Vector3.SignedAngle(Flat(rigid.transform.forward).normalized, dirFlat, Vector3.up);
            float forwardScale = (Mathf.Abs(headingErr) < lookAheadTurnDeg) ? 1.0f : 0.25f;

            float maxStep = speed * forwardScale * Time.fixedDeltaTime;

            float stepLen = Mathf.Min(maxStep, dist);  
            Vector3 nextFlat = posFlat + dirFlat * stepLen;

            // 5) 이동
            rigid.MovePosition(new Vector3(nextFlat.x, rigid.position.y, nextFlat.z));

            UpdateDeltaPose();
            
            yield return new WaitForSeconds(0.05f);

            GameManager.s_agent.StartLidar?.Invoke();
        }
    }

    private void StopMovement()
    {
        wheel1.motorTorque = 0;
        wheel2.motorTorque = 0;
        wheel3.motorTorque = 0;
        wheel4.motorTorque = 0;

        wheel1.steerAngle = 0;
        wheel2.steerAngle = 0;

        rigid.velocity = Vector3.zero;
        rigid.angularVelocity = Vector3.zero;

        
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
        GameManager.s_agent.poseTheta_rad = deltaTheta_rad;

        prevPosition = currPos;
        prevRotationY = currYawDeg;
        
    }
}
