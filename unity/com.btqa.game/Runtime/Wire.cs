// BT-QA Game SDK — 프로토콜 상수·오류·JSON Lines 인코딩. 명세: docs/PROTOCOL.md (Python 판: btqa_game/protocol.py)
using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Btqa.Game
{
    public static class Wire
    {
        /// <summary>1.1 trigger.totals / 1.2 result.stalled / 1.3 classes[].blocks (학습 대상 = 블록 가중치)</summary>
        public const string ProtocolVersion = "1.4";   // 1.4: run_episode.initial_state

        // 학습 → 게임
        public const string Hello = "hello";
        public const string Describe = "describe";
        public const string RunEpisode = "run_episode";
        public const string TriggerResponse = "trigger_response";
        public const string Bye = "bye";

        // 게임 → 학습
        public const string HelloOk = "hello_ok";
        public const string Description = "description";
        public const string Trigger = "trigger";
        public const string EpisodeResult = "episode_result";
        public const string Error = "error";

        // 오류 코드
        public const string EBadRequest = "bad_request";
        public const string EUnknownType = "unknown_type";
        public const string EProtocolVersion = "protocol_version";
        public const string EUnknownClass = "unknown_class";
        public const string EUnknownScenario = "unknown_scenario";
        public const string EUnknownParam = "unknown_param";
        public const string EParamOutOfRange = "param_out_of_range";
        public const string EInternal = "internal";

        public static JObject MakeError(string code, string message, JToken episodeId = null)
        {
            var m = new JObject { ["type"] = Error, ["code"] = code, ["message"] = message };
            if (episodeId != null && episodeId.Type != JTokenType.Null) m["episode_id"] = episodeId.DeepClone();
            return m;
        }

        public static string Major(string version) => (version ?? "").Split('.')[0];
    }

    /// <summary>error 응답으로 바뀌는 예외. 연결은 유지된다.</summary>
    public class ProtocolException : Exception
    {
        public readonly string Code;
        public ProtocolException(string code, string message) : base(message) { Code = code; }
    }

    /// <summary>JSON Lines 한 줄 ↔ JObject.</summary>
    public static class JsonLine
    {
        static readonly JsonSerializerSettings Settings = new JsonSerializerSettings
        {
            DateParseHandling = DateParseHandling.None,   // "2026-..." 문자열을 DateTime 으로 바꾸지 않음
            FloatParseHandling = FloatParseHandling.Double,
        };

        public static JObject Decode(string line)
        {
            JToken token;
            try
            {
                token = JsonConvert.DeserializeObject<JToken>(line, Settings);
            }
            catch (JsonException ex)
            {
                throw new ProtocolException(Wire.EBadRequest, "JSON 파싱 실패: " + ex.Message);
            }
            if (!(token is JObject obj) || obj["type"] == null)
                throw new ProtocolException(Wire.EBadRequest, "메시지는 'type' 을 가진 JSON 객체여야 함");
            return obj;
        }

        /// <summary>한 줄 JSON (개행 없음). 비 ASCII 는 그대로 UTF-8.</summary>
        public static string Encode(JObject msg) => msg.ToString(Formatting.None);
    }

    /// <summary>세션이 쓰는 송수신 통로. Recv 는 연결이 끊기면 null.</summary>
    public interface IMessageChannel
    {
        void Send(JObject msg);
        JObject Recv();
    }
}
