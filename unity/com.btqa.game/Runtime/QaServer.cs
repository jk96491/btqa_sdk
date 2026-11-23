// BT-QA Game SDK — TCP 서버 (Python 판: btqa_game/server.py). 인스턴스 1개 = 포트 1개 = 동시 연결 1개, 요청은 순차 처리.
// 연결이 끊기거나 bye 를 받으면 다음 연결을 기다린다 (프로세스는 종료하지 않음).
using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;

namespace Btqa.Game
{
    public sealed class QaServer
    {
        readonly TcpListener _listener;
        readonly Func<QaSession> _sessionFactory;
        readonly Action<string> _log;
        volatile bool _stopping;
        TcpClient _current;

        // 표시용 통계 (로직과 무관)
        long _connections, _episodes;
        public long Connections => Interlocked.Read(ref _connections);
        public long EpisodesServed => Interlocked.Read(ref _episodes);
        public volatile bool ClientConnected;
        public volatile string LastClass;

        public int Port => ((IPEndPoint)_listener.LocalEndpoint).Port;
        public string Host { get; }

        public QaServer(string host, int port, Func<QaSession> sessionFactory, Action<string> log)
        {
            _sessionFactory = sessionFactory;
            _log = log ?? (_ => { });
            Host = host;
            var addr = host == "localhost" ? IPAddress.Loopback : IPAddress.Parse(host);
            _listener = new TcpListener(addr, port);
        }

        /// <summary>포트를 연다 (port 0 이면 빈 포트). 실패 시 SocketException.</summary>
        public void Start()
        {
            // 대기열 5: 처리 중인 연결이 있으면 다음 연결은 거부되지 않고 순서를 기다린다 (game_sim 과 동일)
            _listener.Start(5);
        }

        /// <summary>연결을 하나씩 받아 끝날 때까지 처리. Stop() 전까지 반환하지 않는다.</summary>
        public void Serve()
        {
            while (!_stopping)
            {
                TcpClient client;
                try
                {
                    client = _listener.AcceptTcpClient();
                }
                catch (Exception ex) when (ex is SocketException || ex is ObjectDisposedException || ex is InvalidOperationException)
                {
                    if (_stopping) return;
                    _log("accept 실패: " + ex.Message);
                    continue;
                }
                _current = client;
                Interlocked.Increment(ref _connections);
                ClientConnected = true;
                try
                {
                    HandleClient(client);
                }
                catch (Exception ex)
                {
                    if (!_stopping) _log("연결 처리 중 오류: " + ex);
                }
                finally
                {
                    ClientConnected = false;
                    _current = null;
                    try { client.Close(); } catch { /* 무시 */ }
                }
            }
        }

        /// <summary>대기·처리 중인 연결을 끊고 Serve() 를 끝낸다 (에디터 Play 종료 등).</summary>
        public void Stop()
        {
            _stopping = true;
            try { _listener.Stop(); } catch { /* 무시 */ }
            try { _current?.Close(); } catch { /* 무시 */ }
        }

        void HandleClient(TcpClient client)
        {
            client.NoDelay = true;
            using (var stream = client.GetStream())
            {
                var ch = new StreamChannel(stream);
                var session = _sessionFactory();
                while (!_stopping)
                {
                    JObject msg;
                    try
                    {
                        msg = ch.Recv();
                    }
                    catch (ProtocolException ex)
                    {
                        ch.Send(Wire.MakeError(ex.Code, ex.Message));
                        continue;
                    }
                    if (msg == null) return;        // 연결 종료
                    bool isEpisode = msg["type"]?.Type == JTokenType.String && (string)msg["type"] == Wire.RunEpisode;
                    if (!session.Handle(msg, ch)) return;   // bye
                    if (isEpisode)
                    {
                        Interlocked.Increment(ref _episodes);
                        LastClass = msg["class"]?.ToString();
                    }
                }
            }
        }

        /// <summary>UTF-8(BOM 없음) JSON 1개 + '\n'.</summary>
        sealed class StreamChannel : IMessageChannel
        {
            readonly StreamReader _reader;
            readonly StreamWriter _writer;

            public StreamChannel(Stream stream)
            {
                var utf8 = new UTF8Encoding(false);
                _reader = new StreamReader(stream, utf8, false, 1 << 16, true);
                _writer = new StreamWriter(stream, utf8, 1 << 16, true) { AutoFlush = false, NewLine = "\n" };
            }

            public void Send(JObject msg)
            {
                _writer.Write(JsonLine.Encode(msg));
                _writer.Write('\n');
                _writer.Flush();
            }

            /// <summary>빈 줄은 건너뜀. 연결 끊김 → null. JSON 오류 → ProtocolException(bad_request).</summary>
            public JObject Recv()
            {
                while (true)
                {
                    string line;
                    try
                    {
                        line = _reader.ReadLine();
                    }
                    catch (Exception ex) when (ex is IOException || ex is ObjectDisposedException)
                    {
                        return null;
                    }
                    if (line == null) return null;
                    if (line.Trim().Length == 0) continue;
                    return JsonLine.Decode(line);
                }
            }
        }
    }
}
