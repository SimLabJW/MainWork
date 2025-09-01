using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Net.Http;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class CommunicationManager
{
    public string Url_result;
    public string Url_Address = "http://210.110.250.32:22665/db/";

    public Action<string, string, List<string>, Dictionary<string, object>, string> OnCommData;
    public Action<string, string, string[]> OnCommSaveData;
    
    public void Communication(string table, List<string> columns, Dictionary<string, object> filters, string commmethod)
    {
        OnCommData?.Invoke(Url_Address, table, columns, filters, commmethod);
    }

    public void SaveCommunication(string table, string[] data)
    {
        OnCommSaveData?.Invoke(Url_Address, table, data);
    }

    //ImportScenario
    public Action<string, string> ScenarioCommAction;

    public void ScenarioButtonCommunication(string scenarioId)
    {
        ScenarioCommAction?.Invoke(Url_Address, scenarioId);
    }

    public Action<string, string,  Dictionary<string, object>, Dictionary<string, object>> ScenarioUpdateAction;
    // UpdateScenarioa
    public void ScenarioUpdate(string table, Dictionary<string, object> data, Dictionary<string, object> filters)
    {
        ScenarioUpdateAction?.Invoke(Url_Address, table, data, filters);
    }

    public Action<string, string, Dictionary<string, object>> ScenarioDeleteAction;
    // UpdateScenarioa
    public void ScenarioDelete(string Table, Dictionary<string, object> filters)
    {
        ScenarioDeleteAction?.Invoke(Url_Address, Table, filters);
    }

    // scenario glb id find, scenario_agent id(colum) find
    public Action<string, string, List<string>, Dictionary<string, object>> ScenarioInfoFindAction;
    public void ScenarioInfoFind(string table, List<string> columns, Dictionary<string, object> filters)
    {
        ScenarioInfoFindAction?.Invoke(Url_Address, table, columns, filters);
    }

    // scenario_agent insert
    public Action<string, string, Dictionary<string, object>> ScenarioAgentInsertAction;
    public void ScenarioAgentInsert(string table, Dictionary<string, object> columns)
    {
        ScenarioAgentInsertAction?.Invoke(Url_Address, table, columns);
    }
}

