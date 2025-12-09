using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CopyAgentMoveUpdate : MonoBehaviour
{

    public Rigidbody rigid;
    public WheelCollider wheel1, wheel2, wheel3, wheel4;
    public float drivespeed = 30.0f, steerspeed;
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
    // Start is called before the first frame update
    void Start()
    {
        GameManager.s_agent.MoveUpdateCopyAgent -= ApplyCommands;
        GameManager.s_agent.MoveUpdateCopyAgent += ApplyCommands;

        GameManager.s_map.RotationCopyAgent -= RotationRobotPose;
        GameManager.s_map.RotationCopyAgent += RotationRobotPose;

    }

    public void ApplyCommands(List<Vector3> waypoints)
    {
        
        StartCoroutine(ExecuteRLCommands(waypoints));
    }

    // y=0으로 납작하게
    Vector3 Flat(Vector3 v) => new Vector3(v.x, 0f, v.z);

    private IEnumerator ExecuteRLCommands(List<Vector3> waypointsFromUnity)
    {
        Debug.Log("IN Apply Copy Command");

        var waypointsWorld = new List<Vector3>(waypointsFromUnity.Count);
        float yKeep = transform.position.y; // 높이 유지
        for (int i = 0; i < waypointsFromUnity.Count; i++)
        {
            Vector3 p = waypointsFromUnity[i];

            float worldX = p.x; // y => X
            float worldZ = p.z; // x => Z

            waypointsWorld.Add(new Vector3(worldX, yKeep, worldZ));
        }

        int startIdx = 0;

        for (int i = startIdx; i < waypointsWorld.Count; i++)
        {
            yield return StartCoroutine(ExecuteSingleCommand(waypointsWorld[i])); // 이동 + 통신/대기 수행
            

        }

        // while (GameManager.s_comm.s_comm_Coroutine != null)
        // {
        //     yield return null;
        // }
        
        yield break;
    } 

    private IEnumerator ExecuteSingleCommand(Vector3 waypointWorld)
    {
        Vector3 targetPosition = new Vector3(waypointWorld.x, transform.position.y, waypointWorld.z);

        // bool isFirstLidarCall = true;
        
        while (true)
        {

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

            
            // transform.position = Vector3.MoveTowards(transform.position, targetPosition, drivespeed * Time.deltaTime);
            
            Vector3 moveTo = Vector3.MoveTowards(transform.position, targetPosition, drivespeed * Time.fixedDeltaTime);
            rigid.MovePosition(moveTo);

            // UpdateRealPose();
            // if (isFirstLidarCall || GameManager.s_comm.s_comm_Coroutine == null)
            // {
            //     isFirstLidarCall = false; 
            //     yield return new WaitForSeconds(0.01f); 
            //     // GameManager.s_agent.StartLidar?.Invoke(); // 진짜 로봇에게 전달.
            // }
            // else
            // {
 
            yield return new WaitForSeconds(0.03f); 
            // }
            
        }
    }

    void RotationRobotPose(string dir)
    {
        Transform agentTransform = GameManager.s_map.CopyAgent.transform;
        float yRotation = 0f;
        if (dir.ToLower() == "left")
        {
            yRotation = -5f;
        }
        else if (dir.ToLower() == "right")
        {
            yRotation = 5f;
        }
        agentTransform.localRotation *= Quaternion.Euler(0f, yRotation, 0f);

        // UpdateRealPose()
    }


    void UpdateRealPose()
    {
        
    }
}
