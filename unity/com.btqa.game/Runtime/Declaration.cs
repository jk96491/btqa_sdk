// BT-QA Game SDK — 정의서(describe 응답) 선언 모델. Python 판: btqa_game/declaration.py
//
// SDK 핵심 규칙 (2026-11-23):  학습 대상 = 전략 선택 노드의 블록 가중치 w (S = N × w) 뿐.
//  - 게임은 BT 의 블록과 가중치 파라미터를 Blocks 로 선언한다.
//  - wire 의 optimize 는 게임이 고르지 않는다. 블록 가중치면 true, 아니면 false 로 SDK 가 정한다.
//  - 가중치가 아닌 BT 값(물약 체력·자원 비축·최소 유지 시간 등)과 페르소나 값(θ, φ)은 기획 default·페르소나가 정한다.
using System;
using System.Collections.Generic;
using System.Globalization;
using Newtonsoft.Json.Linq;

namespace Btqa.Game
{
    public sealed class DeclarationException : Exception
    {
        public readonly List<string> Problems;
        public DeclarationException(List<string> problems) : base("정의서 규칙 위반: " + string.Join("; ", problems))
        {
            Problems = problems;
        }
    }

    public sealed class ParamDecl
    {
        public const string GroupBt = "bt";
        public const string GroupPersona = "persona";
        public const string WeightUnit = "weight";

        public string Name;
        public double Low, High, Default;   // Default = 개발팀 손 튜닝 값 (기준선)
        public string Group = GroupBt;
        public string Unit = "";
        public string Description = "";
        /// <summary>정의서에 optimize 를 직접 적었을 때만 값 (규칙 검사용). null = 적지 않음.</summary>
        public bool? Optimize;

        public bool Contains(double v, double tol = 1e-9) => Low - tol <= v && v <= High + tol;
    }

    public sealed class BlockDecl
    {
        public string Name;
        public string Weight;          // 가중치 파라미터 이름
        public string Description = "";

        public BlockDecl(string name, string weight, string description = "")
        {
            Name = name; Weight = weight; Description = description ?? "";
        }

        public JObject ToWire()
        {
            var o = new JObject { ["name"] = Name, ["weight"] = Weight };
            if (Description.Length > 0) o["description"] = Description;
            return o;
        }
    }

    public sealed class ClassDecl
    {
        public string Name;
        public string DisplayName;
        public bool Ranged;
        public string SpecVersion = "0.1";
        public readonly List<ParamDecl> Params = new List<ParamDecl>();   // 정의서 순서 = describe 순서
        public readonly List<BlockDecl> Blocks = new List<BlockDecl>();

        public ParamDecl Param(string name)
        {
            foreach (var p in Params) if (p.Name == name) return p;
            return null;
        }

        public bool IsWeight(string name)
        {
            foreach (var b in Blocks) if (b.Weight == name) return true;
            return false;
        }

        public List<string> Problems()
        {
            var outp = new List<string>();
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var p in Params)
            {
                if (!names.Add(p.Name)) outp.Add(Name + ": 파라미터 이름 중복 " + p.Name);
                if (p.Group != ParamDecl.GroupBt && p.Group != ParamDecl.GroupPersona)
                    outp.Add(Name + "." + p.Name + ": group 은 bt|persona");
                if (!(p.Low < p.High && p.Low <= p.Default && p.Default <= p.High))
                    outp.Add(Name + "." + p.Name + ": range/default 오류");
                if (!IsWeight(p.Name) && p.Optimize == true)
                    outp.Add(Name + "." + p.Name + ": 가중치가 아닌 파라미터는 학습 대상이 될 수 없음 (optimize=true 금지)");
            }
            if (Blocks.Count == 0) outp.Add(Name + ": blocks 가 비어 있음 (학습할 가중치가 없음)");
            var bnames = new HashSet<string>(StringComparer.Ordinal);
            var weights = new HashSet<string>(StringComparer.Ordinal);
            foreach (var b in Blocks)
            {
                if (!bnames.Add(b.Name)) outp.Add(Name + ": 블록 이름 중복 " + b.Name);
                if (!weights.Add(b.Weight)) outp.Add(Name + ": 블록 두 개가 같은 가중치 " + b.Weight);
                var p = Param(b.Weight);
                if (p == null) { outp.Add(Name + ": 블록 " + b.Name + " 의 가중치 '" + b.Weight + "' 가 params 에 없음"); continue; }
                if (p.Group != ParamDecl.GroupBt) outp.Add(Name + "." + p.Name + ": 가중치는 group bt 여야 함");
                if (p.Low < 0) outp.Add(Name + "." + p.Name + ": 가중치 범위는 0 이상");
                if (p.Unit != "" && p.Unit != ParamDecl.WeightUnit) outp.Add(Name + "." + p.Name + ": 가중치 unit 은 'weight'");
                if (p.Optimize == false) outp.Add(Name + "." + p.Name + ": 블록 가중치를 optimize=false 로 둘 수 없음");
            }
            return outp;
        }

        public JObject ToWire()
        {
            var ps = new JArray();
            foreach (var p in Params)
            {
                bool w = IsWeight(p.Name);
                ps.Add(new JObject
                {
                    ["name"] = p.Name,
                    ["range"] = new JArray(p.Low, p.High),
                    ["default"] = p.Default,
                    ["optimize"] = w,
                    ["group"] = p.Group,
                    ["unit"] = w ? ParamDecl.WeightUnit : p.Unit,
                    ["description"] = p.Description,
                });
            }
            var bs = new JArray();
            foreach (var b in Blocks) bs.Add(b.ToWire());
            return new JObject
            {
                ["name"] = Name,
                ["display_name"] = DisplayName ?? Name,
                ["ranged"] = Ranged,
                ["spec_version"] = SpecVersion,
                ["params"] = ps,
                ["blocks"] = bs,
            };
        }

        /// <summary>
        /// 요청·트리거 응답 params 검증 (부분 덮어쓰기용 값 확인만, 적용은 게임이 한다).
        /// 모르는 이름 unknown_param, 숫자 아님 bad_request, 범위 밖·NaN param_out_of_range.
        /// </summary>
        public void CheckUpdates(JObject updates)
        {
            if (updates == null) return;
            foreach (var prop in updates.Properties())
            {
                var def = Param(prop.Name);
                if (def == null)
                    throw new ProtocolException(Wire.EUnknownParam, "class '" + Name + "' 에 파라미터 '" + prop.Name + "' 없음");
                var t = prop.Value;
                if (t.Type != JTokenType.Float && t.Type != JTokenType.Integer)
                    throw new ProtocolException(Wire.EBadRequest, "파라미터 '" + prop.Name + "' 값이 숫자가 아님: " + t);
                double v = t.Value<double>();
                if (double.IsNaN(v) || double.IsInfinity(v) || !def.Contains(v))
                    throw new ProtocolException(Wire.EParamOutOfRange, string.Format(CultureInfo.InvariantCulture,
                        "{0}={1:G6} 가 범위 [{2:G6}, {3:G6}] 밖", prop.Name, v, def.Low, def.High));
            }
        }
    }

    public sealed class TriggerDecl
    {
        public string Type = "boss_hp_below";            // run_episode.triggers 의 키
        public readonly List<double> Supported = new List<double>();
        public readonly List<string> StateFeatures = new List<string>();   // 정책 입력 순서 (0~1)
        public readonly List<string> Totals = new List<string>();          // trigger.totals 키 (v1.1)

        public JObject ToWire() => new JObject
        {
            ["type"] = Type,
            ["supported"] = new JArray(Supported),
            ["state_features"] = new JArray(StateFeatures),
            ["totals"] = new JArray(Totals),
        };
    }

    public sealed class ScenarioDecl
    {
        public string Name, Description;
        public ScenarioDecl(string name, string description = "") { Name = name; Description = description ?? ""; }
    }

    public sealed class GameDecl
    {
        public string Game, GameVersion;
        public double TimeLimit;
        public int MaxDeaths;
        public double DecisionInterval;
        public readonly List<ClassDecl> Classes = new List<ClassDecl>();
        public readonly List<ScenarioDecl> Scenarios = new List<ScenarioDecl>();
        public TriggerDecl Triggers = new TriggerDecl();
        public readonly List<string> ResultFields = new List<string>();
        /// <summary>게임 고유 추가 필드 (학습 쪽은 모르는 필드를 무시).</summary>
        public JObject Extra;

        public ClassDecl Class(string name)
        {
            if (name == null) return null;
            foreach (var c in Classes) if (c.Name == name) return c;
            return null;
        }

        public bool HasScenario(string name)
        {
            if (name == null) return false;
            foreach (var s in Scenarios) if (s.Name == name) return true;
            return false;
        }

        public List<string> Problems()
        {
            var outp = new List<string>();
            if (Classes.Count == 0) outp.Add("classes 가 비어 있음");
            foreach (var c in Classes) outp.AddRange(c.Problems());
            if (!HasScenario("base")) outp.Add("scenario 'base' 필수");
            if (string.IsNullOrEmpty(Triggers.Type)) outp.Add("triggers.type 필요");
            if (Triggers.StateFeatures.Count == 0) outp.Add("triggers.state_features 비어 있음");
            return outp;
        }

        public GameDecl Validate()
        {
            var p = Problems();
            if (p.Count > 0) throw new DeclarationException(p);
            return this;
        }

        public JObject ToWire()
        {
            var classes = new JArray();
            foreach (var c in Classes) classes.Add(c.ToWire());
            var scenarios = new JArray();
            foreach (var s in Scenarios) scenarios.Add(new JObject { ["name"] = s.Name, ["description"] = s.Description });
            var d = new JObject
            {
                ["type"] = Wire.Description,
                ["protocol_version"] = Wire.ProtocolVersion,
                ["game"] = Game,
                ["game_version"] = GameVersion,
                ["time_limit"] = TimeLimit,
                ["max_deaths"] = MaxDeaths,
                ["decision_interval"] = DecisionInterval,
                ["classes"] = classes,
                ["scenarios"] = scenarios,
                ["triggers"] = Triggers.ToWire(),
            };
            if (ResultFields.Count > 0) d["result_fields"] = new JArray(ResultFields);
            if (Extra != null)
                foreach (var prop in Extra.Properties())
                    if (d[prop.Name] == null) d[prop.Name] = prop.Value.DeepClone();
            return d;
        }
    }
}
