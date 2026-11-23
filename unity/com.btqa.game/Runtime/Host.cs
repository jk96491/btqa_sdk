// BT-QA Game SDK — 게임이 구현하는 확장 지점. Python 판: btqa_game/host.py
//
// 게임 개발자가 할 일은 IEpisodeHost 하나:
//   Declare()    : 정의서 (직업·파라미터·블록 가중치·시나리오·트리거)
//   RunEpisode() : 판 1개를 seed 로 결정론적으로 끝까지. 트리거 경계마다 triggers.Fire(...) (응답까지 게임 시간 정지)
// 요청 검증·트리거 왕복 검사·오류 응답·연결 유지는 SDK(QaSession) 가 한다.
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;

namespace Btqa.Game
{
    /// <summary>검증을 마친 run_episode 요청.</summary>
    public sealed class EpisodeRequest
    {
        public JToken EpisodeId;
        public ClassDecl Class;
        public string Scenario;
        public long Seed;
        public JToken SeedToken;          // 요청 seed 원본 (result.seed 에 그대로 돌려줄 때)
        public JObject Params;            // 요청 params (검증 완료, null 가능). 생략한 값 = 정의서 default (게임이 적용)
        public List<double> Thresholds;   // 요청 트리거 경계 (빈 목록 = 트리거 없음)
        public bool RecordEvents;
        public JObject Raw;               // 원본 메시지
        /// <summary>v1.4 판 시작 상태: 선언 전체(default + 요청, 검증 완료). null = 요청 없음 (1.3 과 같은 시작).</summary>
        public Dictionary<string, double> InitialState;
    }

    public interface IEpisodeHost
    {
        /// <summary>정의서. SDK 가 Validate() 후 describe 응답으로 보낸다.</summary>
        GameDecl Declare();

        /// <summary>판 1개 → episode_result.result (docs/PROTOCOL.md 4.5).</summary>
        JObject RunEpisode(EpisodeRequest request, TriggerChannel triggers);
    }

    /// <summary>판 진행 중 trigger ↔ trigger_response 왕복.</summary>
    public sealed class TriggerChannel
    {
        readonly EpisodeRequest _req;
        readonly IMessageChannel _ch;
        public int Fired { get; private set; }

        public TriggerChannel(EpisodeRequest req, IMessageChannel ch) { _req = req; _ch = ch; }

        /// <summary>trigger 를 보내고 응답을 기다린다. 반환 = 덮어쓸 params (검증 완료, null 가능).</summary>
        public JObject Fire(int index, JObject state, JObject totals)
        {
            _ch.Send(new JObject
            {
                ["type"] = Wire.Trigger,
                ["episode_id"] = _req.EpisodeId.DeepClone(),
                ["index"] = index,
                ["state"] = state,
                ["totals"] = totals,
            });
            var reply = _ch.Recv();   // 응답이 올 때까지 판(게임 시간) 정지
            if (reply == null) throw new IOException("trigger 응답 대기 중 연결 종료");
            var po = CheckResponse(reply, "episode_id", _req.EpisodeId, index);
            _req.Class.CheckUpdates(po);
            Fired++;
            return po;
        }

        /// <summary>
        /// trigger_response 형식 검사 → params (null 가능). idKey = episode_id(밸런스) / encounter_id(플레이) 등.
        /// 틀리면 ProtocolException(bad_request).
        /// </summary>
        public static JObject CheckResponse(JObject reply, string idKey, JToken idValue, int index)
        {
            var type = reply["type"]?.Type == JTokenType.String ? (string)reply["type"] : null;
            if (type != Wire.TriggerResponse || !JToken.DeepEquals(reply[idKey], idValue))
                throw new ProtocolException(Wire.EBadRequest,
                    "trigger_response(" + idKey + "=" + idValue + ") 를 기대했으나 type=" + reply["type"] + " " + idKey + "=" + reply[idKey]);
            var idx = reply["index"];
            if (idx == null || (idx.Type != JTokenType.Integer && idx.Type != JTokenType.Float) || idx.Value<double>() != index)
                throw new ProtocolException(Wire.EBadRequest, "trigger_response index " + idx + " != " + index);
            var ps = reply["params"];
            if (ps == null || ps.Type == JTokenType.Null) return null;
            if (!(ps is JObject po)) throw new ProtocolException(Wire.EBadRequest, "trigger_response.params 는 객체여야 함");
            return po;
        }

        /// <summary>triggers.&lt;type&gt; 숫자 배열 파싱. 없으면 빈 목록.</summary>
        public static List<double> ParseThresholds(JToken t, string type)
        {
            var outp = new List<double>();
            if (t == null || t.Type == JTokenType.Null) return outp;
            if (!(t is JObject o)) throw new ProtocolException(Wire.EBadRequest, "triggers 는 객체여야 함");
            var b = o[type];
            if (b == null || b.Type == JTokenType.Null) return outp;
            if (!(b is JArray arr)) throw new ProtocolException(Wire.EBadRequest, "triggers." + type + " 는 숫자 배열이어야 함");
            foreach (var x in arr)
            {
                if (x.Type != JTokenType.Float && x.Type != JTokenType.Integer)
                    throw new ProtocolException(Wire.EBadRequest, "triggers." + type + " 항목은 숫자");
                outp.Add(x.Value<double>());
            }
            return outp;
        }
    }
}
