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
    public string Url_Address = "http://210.110.250.32:28000/db/";

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

}

// [System.Serializable]
// public class Wrapper
// {
//     public string table;
//     public List<string> columns;
//     public Dictionary<string, object> filters;

    
// }