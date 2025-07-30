using System.Collections.Generic;
using UnityEngine;

public class ObjectProfile
{
    [System.Serializable]
    public class SelectTagProfile
    {
        public string ObjectTag;
     }

    [System.Serializable]
    public class MapObjectProfile
    {
        public string MapName;
        
        public float Map_x;
        public float Map_y;
        public float Map_z;

        // ObjectListView 로 보여주기
        public List<GameObject> PlaneObject;
        public List<GameObject> ObstacleObject;
    }

    [System.Serializable]
    public class AgentObjectProfile
    {
        public string AgentName;

        public float Agent_x;
        public float Agent_y;
        public float Agent_z;

        // ObjectListView 로 보여주기
        public List<GameObject> WheelObject;
        public List<GameObject> FloaterObject;
        public List<GameObject> SensorObject;
        public List<GameObject> WeaponeObject;
    }

    // 특정 번호로 변경 가능
    public void ObjectTagProfileApply(GameObject go)
    {
        var info = new SelectTagProfile
        {
            ObjectTag = go.tag,
        };
    }

    public void MapProfileInputFieldApply(GameObject go)
    {
        var profile = GameManager.simulation.sm.simulationInfo.editorInform.env_profile;

        var map_info = new MapObjectProfile
        {
            MapName = go.name,

            Map_x = go.transform.position.x,
            Map_y = go.transform.position.y,
            Map_z = go.transform.position.z
        };


        profile.EnvName.text = map_info.MapName;
        profile.Env_x.text = map_info.Map_x.ToString();
        profile.Env_y.text = map_info.Map_y.ToString();
        profile.Env_z.text = map_info.Map_z.ToString();


        return;
    }

    public MapObjectProfile MapProfileButtonApply(GameObject go)
    {
        var profile = GameManager.simulation.sm.simulationInfo.editorInform.env_profile;

        var map_info = new MapObjectProfile();
    
        Transform Planes = go.transform.Find("Planes");
        if (Planes != null)
        {
            map_info.PlaneObject = new List<GameObject>();

            foreach (Transform child in Planes)
            {
                map_info.PlaneObject.Add(child.gameObject);
            }
        }

        Transform Obstacles = go.transform.Find("Obstacles");
        if (Obstacles != null)
        {
            map_info.ObstacleObject = new List<GameObject>();

            foreach (Transform child in Obstacles)
            {
                map_info.ObstacleObject.Add(child.gameObject);
            }
        }

        return map_info;
    }

    public void AgentProfileInputFielApply(GameObject go)
    {
        var profile = GameManager.simulation.sm.simulationInfo.editorInform.agent_profile;

        var agent_info = new AgentObjectProfile
        {
            AgentName = go.name,

            Agent_x = go.transform.position.x,
            Agent_y = go.transform.position.y,
            Agent_z = go.transform.position.z,
        };

        profile.AgentName.text = agent_info.AgentName;
        profile.Agent_x.text = agent_info.Agent_x.ToString();
        profile.Agent_y.text = agent_info.Agent_y.ToString();
        profile.Agent_z.text = agent_info.Agent_z.ToString();

    }

    public AgentObjectProfile AgentProfileButtonApply(GameObject go)
    {
        var profile = GameManager.simulation.sm.simulationInfo.editorInform.agent_profile;

        var agent_info = new AgentObjectProfile();

        GameObject goChild = go.transform.GetChild(0).gameObject;

        Transform Wheels = goChild.transform.Find("Wheels");
        if (Wheels != null)
        {
            agent_info.WheelObject = new List<GameObject>();

            foreach (Transform child in Wheels)
            {
                agent_info.WheelObject.Add(child.gameObject);
            }
        }

        Transform Floaters = goChild.transform.Find("Floaters");
        if (Floaters != null)
        {
            agent_info.FloaterObject = new List<GameObject>();

            foreach (Transform child in Floaters)
            {
                agent_info.FloaterObject.Add(child.gameObject);
            }
        }
        Transform Sensors = goChild.transform.Find("Sensors");
        if (Sensors != null)
        {
            agent_info.SensorObject = new List<GameObject>();

            foreach (Transform child in Sensors)
            {
                agent_info.SensorObject.Add(child.gameObject);
            }
        }

        Transform Weapones = goChild.transform.Find("Weapones");
        if (Weapones != null)
        {
            agent_info.WeaponeObject = new List<GameObject>();

            foreach (Transform child in Weapones)
            {
                agent_info.WeaponeObject.Add(child.gameObject);
            }
        }

        return agent_info;
        
    }
}
