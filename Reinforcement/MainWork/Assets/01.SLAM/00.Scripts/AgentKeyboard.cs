using System.Collections;
using UnityEngine;

public class AgentKeyboard : MonoBehaviour
{
    public Rigidbody rigid;
    public WheelCollider wheel1, wheel2, wheel3, wheel4;
    public float drivespeed,steerspeed;
    public float rotationSpeed = 30f;  // Q/E 회전 속도 (deg/s)

    private Coroutine controlCoroutine;

    private Vector3 prevPosition;
    private float prevRotationY;

    void Start()
    {
        prevPosition = transform.position;
        prevRotationY = transform.eulerAngles.y;

        controlCoroutine = StartCoroutine(KeyboardControlLoop());
    }

    private IEnumerator KeyboardControlLoop()
    {
        while (true)
        {
            float motor = 0f;
            float steer = 0f;
            float rotate = 0f;

            // --- 키 입력 ---
            if (Input.GetKey(KeyCode.W)) motor = drivespeed;   // 전진
            if (Input.GetKey(KeyCode.S)) motor = -drivespeed;  // 후진
            if (Input.GetKey(KeyCode.A)) steer = -steerspeed;  // 좌측 조향
            if (Input.GetKey(KeyCode.D)) steer = steerspeed;   // 우측 조향

            if (Input.GetKey(KeyCode.Q)) rotate = -1f;         // 좌로 제자리 회전
            if (Input.GetKey(KeyCode.E)) rotate = 1f;          // 우로 제자리 회전

            // --- 바퀴 구동/조향 적용 ---
            wheel1.motorTorque = motor;
            wheel2.motorTorque = motor;
            wheel3.motorTorque = motor;
            wheel4.motorTorque = motor;

            wheel1.steerAngle = steer;
            wheel2.steerAngle = steer;

            // --- 제자리 회전 ---
            if (rotate != 0f)
            {
                transform.Rotate(Vector3.up * rotate * rotationSpeed * Time.deltaTime);
            }

            // --- Δpose 업데이트 ---
            UpdateDeltaPose();

            // 키보드 입력 없을 때 정지 처리
            if (motor == 0f && steer == 0f && rotate == 0f)
            {
                StopMovement();
            }

            yield return null;
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

        // GameManager에 기록
        GameManager.s_agent.deltaX_m = deltaX_m;
        GameManager.s_agent.deltaY_m = deltaY_m;
        GameManager.s_agent.deltaTheta_rad = deltaTheta_rad;

        prevPosition = currPos;
        prevRotationY = currYawDeg;
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
}
