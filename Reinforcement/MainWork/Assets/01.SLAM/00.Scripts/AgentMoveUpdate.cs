using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentAnimation : MonoBehaviour
{
    public Rigidbody rigid;
    public WheelCollider wheel1, wheel2, wheel3, wheel4;
    public float drivespeed, steerspeed;
    public float rotationSpeed = 30.0f;

    private Coroutine moveCoroutine;
    private Coroutine TrainDoneCoroutine;

    public void Start()
    {
        // TestGameManager.Test_Agent.OnMoveTraining -= UpdateMovementData;
        // TestGameManager.Test_Agent.OnMoveTraining += UpdateMovementData; 

        // TestGameManager.Test_Agent.Reset_device -= Reset_Agent;
        // TestGameManager.Test_Agent.Reset_device += Reset_Agent;

        // TestGameManager.Test_Agent.OnMoveTrainingDone -= ApplyMovementData;
        // TestGameManager.Test_Agent.OnMoveTrainingDone += ApplyMovementData;

    }

    public void UpdateMovementData(int nx, int nz, int angle)
    {
        // 새로운 이동 코루틴이 실행 중이라면 중지하고 새로 시작
        if (moveCoroutine != null)
        {
            StopCoroutine(moveCoroutine);
        }

        // 에이전트에 주어진 위치로 즉시 이동
        moveCoroutine = StartCoroutine(ProcessMovement(nx, nz, angle));
    }

    private IEnumerator ProcessMovement(int targetX, int targetZ, int targetAngle)
    {
        yield return StartCoroutine(RotateAgent(targetAngle));
        yield return StartCoroutine(MoveAgent(targetX, targetZ));
        FinalizeMovement();
    }

    // 에이전트 회전 처리
    private IEnumerator RotateAgent(float targetAngle)
    {
        // 각도 제한 (유효 범위 내로)
        targetAngle = Mathf.Clamp(targetAngle, -90, 180);
        transform.eulerAngles = new Vector3(0, targetAngle, 0);
        yield return null;  // 회전 완료 후
    }

    // 에이전트 위치 이동 처리
    private IEnumerator MoveAgent(int targetX, int targetZ)
    {
        Vector3 newPosition = new Vector3(targetX, transform.position.y, targetZ);
        transform.position = newPosition;
        yield return null;  // 이동 완료 후
    }

    // 이동 완료 처리
    private void FinalizeMovement()
    {
        // TestGameManager.Test_Agent.agent_position_x = transform.position.x;
        // TestGameManager.Test_Agent.agent_position_z = transform.position.z;
        // TestGameManager.Test_Agent.agent_angle = transform.eulerAngles.y;

        // TestGameManager.Test_Agent.OnMovementComplete?.Invoke();
    }


    public void ApplyMovementData(int targetX, int targetZ, int targetAngle)
    {
        if (TrainDoneCoroutine != null)
        {
            StopCoroutine(TrainDoneCoroutine);
        }
        TrainDoneCoroutine = StartCoroutine(MoveForPosition(targetX, targetZ, targetAngle));
    }

    private IEnumerator MoveForPosition(int targetX, int targetZ, int targetAngle)
    {
        Vector3 startPosition = transform.position;
        Vector3 targetPosition = new Vector3(targetX, transform.position.y, targetZ);

        float startAngle = transform.eulerAngles.y;
        float targetRotationAngle = targetAngle;

        bool rotationComplete = false;
        bool distanceComplete = false;

        // 회전 애니메이션 처리
        while (!rotationComplete)
        {
            float currentAngle = Mathf.MoveTowardsAngle(transform.eulerAngles.y, targetRotationAngle, rotationSpeed * Time.deltaTime);
            transform.eulerAngles = new Vector3(0, currentAngle, 0);

            // 회전 완료 여부 체크
            if (Mathf.Abs(currentAngle - targetRotationAngle) < 0.1f)
            {
                rotationComplete = true;
            }
            yield return null; // 한 프레임 대기
        }

        // 위치 이동
        while (!distanceComplete)
        {
            transform.position = Vector3.MoveTowards(transform.position, targetPosition, drivespeed * Time.deltaTime);

            // 이동 완료 여부 체크
            if (Vector3.Distance(transform.position, targetPosition) < 0.1f)
            {
                distanceComplete = true;
            }
            yield return null; // 한 프레임 대기
        }

        StopMovement(); // 이동 정지
        TrainDoneCoroutine = null;
    }

    private void StopMovement()
    {
        // 모든 바퀴 토크와 조향을 0으로 설정하여 이동 정지
        wheel1.motorTorque = 0;
        wheel2.motorTorque = 0;
        wheel3.motorTorque = 0;
        wheel4.motorTorque = 0;
        wheel1.steerAngle = 0;
        wheel2.steerAngle = 0;

        // Rigidbody의 속도와 각속도를 0으로 설정하여 완전 정지
        rigid.velocity = Vector3.zero;
        rigid.angularVelocity = Vector3.zero;

        // TestGameManager.Test_Agent.agent_position_x = transform.position.x;
        // TestGameManager.Test_Agent.agent_position_z = transform.position.z;
        // TestGameManager.Test_Agent.agent_angle = transform.eulerAngles.y;

        // TestGameManager.Test_Agent.OnMovementComplete?.Invoke();
    }



    // private void Reset_Agent(Vector3 position, float angle)
    // {
    //     // 에이전트 위치와 회전을 새로운 값으로 초기화
    //     transform.position = position;
    //     transform.rotation = Quaternion.Euler(0, angle, 0);

    //     TestGameManager.Test_Agent.agent_position_x = transform.position.x;
    //     TestGameManager.Test_Agent.agent_position_z = transform.position.z;
    //     TestGameManager.Test_Agent.agent_angle = transform.rotation.y;
  
    // }
}
