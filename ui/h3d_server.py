# -*- coding: utf-8 -*-
"""h3-director 可视化前端服务器（只依赖标准库）

启动:
  python ui/h3d_server.py                 # http://127.0.0.1:8190
  python ui/h3d_server.py --port 9000
  python ui/h3d_server.py --host 0.0.0.0  # 局域网可访问（注意这是无鉴权服务）

UI 在同目录 index.html。生成任务通过子进程调 h3d.py，日志实时回传页面。
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 本文件在 ui/ 里
sys.path.insert(0, ROOT)
import h3core as H3          # noqa: E402

OUT_ROOT = os.path.join(H3.OUTPUT_DIR, "h3d")
REF_POOL = os.path.join(ROOT, "ref_pool")
os.makedirs(REF_POOL, exist_ok=True)

PY = sys.executable
H3D = os.path.join(ROOT, "h3d.py")

# 运行状态（单任务：GPU 一次只能跑一个）
JOB = {"running": False, "started": 0, "log": [], "script": "", "name": ""}
JOB_LOCK = threading.RLock()   # 可重入：start_job 持锁时内部还要 log_line


def log_line(s):
    with JOB_LOCK:
        JOB["log"].append(f"[{time.strftime('%H:%M:%S')}] {s}")
        JOB["log"] = JOB["log"][-400:]


def start_job(meta, segs):
    with JOB_LOCK:
        if JOB["running"]:
            return False, "已有任务在跑"
        if not segs:
            return False, "分镜是空的"
        ts = time.strftime("%m%d_%H%M%S")
        name = meta.get("out") or f"h3d_{ts}"
        script = os.path.join(ROOT, f"_webjob_{ts}.txt")
        lines = [f"@out {name}", f"@tier {meta.get('tier', '中')}"]
        for key, tag in (("refs", "@refs "), ("audio", "@audio "),
                         ("negative", "@negative "), ("seed", "@seed "),
                         ("wide", "@wide "), ("res", "@res ")):
            v = meta.get(key)
            if not v:
                continue
            lines.append(tag + (" ".join(v) if isinstance(v, list) else str(v)))
        lines.append("---")
        lines += segs
        with open(script, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        JOB.update(running=True, started=time.time(), log=[], script=script, name=name)
        log_line(f"任务启动: {name} ({meta.get('tier')} / {len(segs)} 段)")

        def worker():
            # close_fds：Windows 上不然 HTTP socket 会被子进程继承，客户端收不到响应
            kw = {}
            if os.name == "nt":
                kw = dict(close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW)
            p = subprocess.Popen(
                [PY, H3D, "run", "--script", script, "--name", name],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", **kw)
            for line in p.stdout:
                log_line(line.rstrip())
            p.wait()
            with JOB_LOCK:
                JOB["running"] = False
                log_line("任务结束" if p.returncode == 0 else f"任务失败 exit={p.returncode}")

        threading.Thread(target=worker, daemon=True).start()
        return True, name


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, open(os.path.join(ROOT, "ui", "index.html"), "rb").read(),
                       "text/html; charset=utf-8")
        elif u.path == "/api/status":
            with JOB_LOCK:
                self._json({"running": JOB["running"], "log": JOB["log"][-60:],
                            "name": JOB["name"],
                            "elapsed": round(time.time() - JOB["started"]) if JOB["running"] else 0})
        elif u.path == "/api/videos":
            items = []
            for root, _dirs, files in os.walk(OUT_ROOT):
                for f in files:
                    if not f.endswith(".mp4") or f.startswith("_"):
                        continue
                    fp = os.path.join(root, f)
                    items.append({"path": os.path.relpath(fp, OUT_ROOT).replace("\\", "/"),
                                  "name": f, "mtime": os.path.getmtime(fp),
                                  "size": os.path.getsize(fp)})
            items.sort(key=lambda x: -x["mtime"])
            self._json({"videos": items[:40]})
        elif u.path == "/api/media":
            rel = unquote(parse_qs(u.query).get("path", [""])[0])
            base = os.path.normpath(OUT_ROOT)
            fp = os.path.normpath(os.path.join(base, rel))
            # 目录穿越防护：只允许读成片目录里的文件
            if not fp.startswith(base + os.sep) or not os.path.isfile(fp):
                self._json({"error": "not found"}, 404)
                return
            self._send(200, open(fp, "rb").read(), "video/mp4")
        elif u.path == "/api/stats":
            pj = os.path.join(ROOT, "state", "params.json")
            self._json(json.load(open(pj, encoding="utf-8")) if os.path.exists(pj)
                       else {"tiers": {}})
        elif u.path == "/api/lessons":
            lm = os.path.join(ROOT, "state", "lessons.md")
            text = open(lm, encoding="utf-8").read() if os.path.exists(lm) else ""
            self._json({"lessons": "\n".join(text.splitlines()[-40:])})
        else:
            self._json({"error": "unknown"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json({"ok": False, "msg": "请求体不是合法 JSON"}, 400)
            return
        u = urlparse(self.path)
        if u.path == "/api/start":
            ok, msg = start_job(body.get("meta", {}), body.get("segs", []))
            self._json({"ok": ok, "msg": msg})
        elif u.path == "/api/upload":
            name = os.path.basename(body.get("name", "img.png"))
            try:
                data = base64.b64decode(body.get("data", ""), validate=True)
            except Exception:
                self._json({"ok": False, "msg": "图片数据解不开"}, 400)
                return
            dst = os.path.join(REF_POOL, name)
            with open(dst, "wb") as f:
                f.write(data)
            # 给绝对路径：h3d 对绝对路径直接用，跨盘 relpath 在 Windows 上会炸
            self._json({"ok": True, "path": dst.replace("\\", "/")})
        elif u.path == "/api/ping":
            self._json({"comfy": H3.comfy_alive(), "url": H3.COMFY,
                        "comfy_dir": bool(H3.COMFY_DIR)})
        else:
            self._json({"error": "unknown"}, 404)

    def log_message(self, *_):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"h3-director 导演台: http://{a.host}:{a.port}  (Ctrl+C 停)")
    print(f"ComfyUI: {H3.COMFY}  目录: {H3.COMFY_DIR or '(未配置，去改 config.json)'}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
