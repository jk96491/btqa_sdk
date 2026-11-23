// BT-QA Game SDK — 연결 1개에 대한 프로토콜 세션. Python 판: btqa_game/session.py
//   hello → hello_ok, describe → description,
//   run_episode → (trigger ↔ trigger_response)* → episode_result, bye → 종료
using System;
using System.IO;
using Newtonsoft.Json.Linq;

namespace Btqa.Game
{
    public sealed class QaSession
    {
        readonly IEpisodeHost _host;
        GameDecl _decl;

        public QaSession(IEpisodeHost host) { _host = host; }

        public GameDecl Decl => _decl ?? (_decl = _host.Declare().Validate());

        // ------------------------------------------------------------------ 요청 검증
        public EpisodeRequest ParseRequest(JObject msg)
        {
            foreach (var key in new[] { "episode_id", "class", "seed" })
                if (msg[key] == null) throw new ProtocolException(Wire.EBadRequest, "run_episode 에 '" + key + "' 필요");
            var cls = Decl.Class(msg["class"].Type == JTokenType.String ? (string)msg["class"] : null);
            if (cls == null) throw new ProtocolException(Wire.EUnknownClass, "알 수 없는 class " + msg["class"]);
            var scTok = msg["scenario"];
            var scenario = scTok == null || scTok.Type == JTokenType.Null ? "base" : scTok.Type == JTokenType.String ? (string)scTok : null;
            if (scenario == "") scenario = "base";
            if (!Decl.HasScenario(scenario)) throw new ProtocolException(Wire.EUnknownScenario, "알 수 없는 scenario " + scTok);
            var ps = msg["params"];
            if (ps != null && ps.Type != JTokenType.Null && ps.Type != JTokenType.Object)
                throw new ProtocolException(Wire.EBadRequest, "params 는 객체여야 함");
            var seedTok = msg["seed"];
            if (seedTok.Type != JTokenType.Integer) throw new ProtocolException(Wire.EBadRequest, "seed 는 정수여야 함");
            long seed;
            try { seed = seedTok.Value<long>(); }
            catch (Exception) { throw new ProtocolException(Wire.EBadRequest, "seed 는 64비트 정수여야 함"); }
            var thresholds = TriggerChannel.ParseThresholds(msg["triggers"], Decl.Triggers.Type);
            var po = ps as JObject;
            cls.CheckUpdates(po);
            return new EpisodeRequest
            {
                EpisodeId = msg["episode_id"],
                Class = cls,
                Scenario = scenario,
                Seed = seed,
                SeedToken = seedTok,
                Params = po,
                Thresholds = thresholds,
                RecordEvents = msg["record_events"]?.Type == JTokenType.Boolean && (bool)msg["record_events"],
                Raw = msg,
            };
        }

        JObject RunEpisode(JObject msg, IMessageChannel ch)
        {
            var req = ParseRequest(msg);
            var result = _host.RunEpisode(req, new TriggerChannel(req, ch));
            return new JObject
            {
                ["type"] = Wire.EpisodeResult,
                ["episode_id"] = req.EpisodeId.DeepClone(),
                ["result"] = result,
            };
        }

        // ------------------------------------------------------------------ 디스패치
        /// <summary>메시지 1개 처리. 연결을 닫아야 하면 false.</summary>
        public bool Handle(JObject msg, IMessageChannel ch)
        {
            var kind = msg["type"]?.Type == JTokenType.String ? (string)msg["type"] : null;
            try
            {
                switch (kind)
                {
                    case Wire.Hello:
                    {
                        var v = msg["protocol_version"]?.ToString() ?? "";
                        if (Wire.Major(v) != Wire.Major(Wire.ProtocolVersion))
                            throw new ProtocolException(Wire.EProtocolVersion, "지원 버전 " + Wire.ProtocolVersion + ", 요청 '" + v + "'");
                        ch.Send(new JObject
                        {
                            ["type"] = Wire.HelloOk,
                            ["protocol_version"] = Wire.ProtocolVersion,
                            ["game"] = Decl.Game,
                            ["game_version"] = Decl.GameVersion,
                        });
                        break;
                    }
                    case Wire.Describe:
                        ch.Send(Decl.ToWire());
                        break;
                    case Wire.RunEpisode:
                        ch.Send(RunEpisode(msg, ch));
                        break;
                    case Wire.Bye:
                        return false;
                    default:
                        throw new ProtocolException(Wire.EUnknownType, "알 수 없는 type '" + msg["type"] + "'");
                }
            }
            catch (ProtocolException ex)
            {
                ch.Send(Wire.MakeError(ex.Code, ex.Message, msg["episode_id"]));
            }
            catch (IOException)
            {
                return false;   // 연결 끊김 (트리거 응답 대기 중 등)
            }
            catch (Exception ex)  // 게임 내부 오류도 연결은 유지
            {
                ch.Send(Wire.MakeError(Wire.EInternal, ex.GetType().Name + ": " + ex.Message, msg["episode_id"]));
            }
            return true;
        }
    }
}
