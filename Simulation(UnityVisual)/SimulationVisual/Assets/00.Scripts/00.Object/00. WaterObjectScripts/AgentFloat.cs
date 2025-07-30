using UnityEngine;
using UnityEngine.Rendering.HighDefinition;
using System.Collections;

public class AgentFloat : MonoBehaviour
{
    public Rigidbody rb;
    public float depthBefSub = 0.8f;
    public float displacementAmt = 1f;
    public float waterDrag = 1f;
    public float waterAngularDrag = 1f;
    public WaterSurface water;

    private WaterSearchParameters Search;
    private WaterSearchResult SearchResult;

    private void OnEnable()
    {
        // 코루틴 시작
        StartCoroutine(ApplyBuoyancy());
    }

    private IEnumerator ApplyBuoyancy()
    {
        WaitForFixedUpdate wait = new WaitForFixedUpdate(); // 물리 프레임 기준

        while (true)
        {
            if (rb == null || water == null)
            {
                yield return wait;
                continue;
            }

            rb.AddForceAtPosition(Physics.gravity / 4f, transform.position, ForceMode.Force);

            Search.startPositionWS = transform.position;
            water.ProjectPointOnWaterSurface(Search, out SearchResult);

            if (transform.position.y < SearchResult.projectedPositionWS.y)
            {
                float displacementMulti = Mathf.Clamp01(
                    (SearchResult.projectedPositionWS.y - transform.position.y) / depthBefSub) * displacementAmt;

                // 부력 적용
                Vector3 buoyancy = Vector3.up * Mathf.Abs(Physics.gravity.y) * displacementMulti;
                rb.AddForceAtPosition(buoyancy, transform.position, ForceMode.Force);

                // 저항 적용
                Vector3 damping = -rb.velocity * waterDrag;
                rb.AddForce(damping * displacementMulti * Time.fixedDeltaTime, ForceMode.VelocityChange);

                rb.AddTorque(
                    displacementMulti * -rb.angularVelocity * waterAngularDrag * Time.fixedDeltaTime,
                    ForceMode.VelocityChange
                );
            }

            yield return wait;
        }
    }
}
