# -*- coding: utf-8 -*-
"""h3core —— MiniMax H3 / ComfyUI 最小客户端（h3-director 自带，无外部依赖）

只做四件事，不掺业务逻辑：
  1. 把 ComfyUI 前端工作流 JSON 翻成 API prompt
  2. 按档位拼 payload（文生 / 首帧接力 / 多参考图 / LoRA / 多级精修）
  3. 提交 + 轮询 + 取结果
  4. token 预算换算（决定 8GB 卡能跑多大）

配置来自仓库根 config.json，环境变量优先，见 load_config()。

依赖：Python 3.9+ 标准库。不需要 requests。
"""
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# 本机常开代理会把 127.0.0.1 也送进代理，导致连不上 ComfyUI。
# 在导入阶段就摘掉，比每个请求单独设 no_proxy 可靠。
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

DEFAULTS = {
    "comfy_url": "http://127.0.0.1:8188",
    "comfy_dir": "",
    "output_dir": "output",
    "workflow": "workflows/h3_base_124f.json",
    "lora_name": "pika_h3_flex_turbo_4step_fl2va_lite_r32.safetensors",
    "latent_upscaler": "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    "ffmpeg": "ffmpeg",
    "ffprobe": "ffprobe",
}


def load_config(path=None):
    """读 config.json（缺省在仓库根），环境变量覆盖关键项。

    环境变量：
      H3_COMFY_URL    ComfyUI 地址
      H3_COMFY_DIR    ComfyUI 根目录（必须能访问到 input/ 和 output/）
      H3_OUTPUT_DIR   成片输出目录
      H3_LORA         4 步加速 LoRA 文件名，设空字符串则不用 LoRA
    """
    cfg = dict(DEFAULTS)
    p = path or os.path.join(HERE, "config.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            cfg.update(json.load(f))
    env_map = {"H3_COMFY_URL": "comfy_url", "H3_COMFY_DIR": "comfy_dir",
               "H3_OUTPUT_DIR": "output_dir", "H3_LORA": "lora_name"}
    for k, field in env_map.items():
        if k in os.environ:
            cfg[field] = os.environ[k]
    # 相对路径一律按仓库根解析，避免从别处调用时找不到文件
    for field in ("workflow", "output_dir"):
        v = cfg.get(field) or ""
        if v and not os.path.isabs(v):
            cfg[field] = os.path.join(HERE, v)
    return cfg


CFG = load_config()
COMFY = str(CFG["comfy_url"]).rstrip("/")
COMFY_DIR = CFG.get("comfy_dir") or ""
COMFY_INPUT = os.path.join(COMFY_DIR, "input") if COMFY_DIR else ""
COMFY_OUTPUT = os.path.join(COMFY_DIR, "output") if COMFY_DIR else ""
LORA_NAME = CFG.get("lora_name") or ""
LATENT_UPSCALER = CFG.get("latent_upscaler") or ""
WORKFLOW = CFG["workflow"]
OUTPUT_DIR = CFG["output_dir"]

# 单次提交的 token 硬上限。8GB 卡实测 133,200（1280x704x124）能过，
# 再高就 OOM。显存更大的卡可以往上调，但 ComfyUI 侧还有一层模型自带的限制。
TOKEN_LIMIT = int(os.environ.get("H3_TOKEN_LIMIT", "133200"))


# ------------------------------------------------------------------ HTTP

def http_json(url, data=None, timeout=60):
    """极简 JSON 请求。data=None 走 GET，否则 POST。"""
    if data is None:
        r = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(r.read().decode("utf-8"))
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError("HTTP %s: %s" % (e.code, detail[:2000]))
    return json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------- token 预算

def latent_t(frames):
    """帧数 -> H3 的 latent 时间步数。H3 的时间压缩是 17 帧 -> 5 步的节奏。"""
    return ((int(frames) - 5) // 17) * 5 + 2


def tokens(w, h, frames):
    """一次采样要吃多少 token。显存够不够看这个数，别看像素。"""
    return (w // 16) * (h // 16) * latent_t(frames)


def fits(w, h, frames, limit=None):
    return tokens(w, h, frames) <= (limit or TOKEN_LIMIT)


# ---------------------------------------------- 前端工作流 JSON -> API prompt

# 纯前端节点，提交会被 ComfyUI 拒绝，直接剔除
SKIP = {"Note", "MarkdownNote", "NoteNode"}

# 连线输入类型（tensor）。除这些之外都当成 widget 回填
LINK_TYPES = {"MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
              "AUDIO", "VIDEO", "CONTROL_NET", "CLIP_VISION", "STYLE_MODEL",
              "GLIGEN", "UPSCALE_MODEL", "NOISE", "GUIDER", "SAMPLER", "SIGMAS",
              "WEIGHTS", "TIMESTEPS", "REGIONAL_ATTENTION", "MODEL_PATCHER"}


def is_widget(spec):
    """object_info 里的一项是不是 widget（而不是连线输入）。

    下拉框在各版本里叫 COMBO / COMFY_DYNAMICCOMBO_V3 / 旧式 [["a","b"],{}]，
    所以用排除法（不是连线类型就算 widget）比穷举类型名稳。
    """
    if not isinstance(spec, list) or not spec:
        return False
    t = spec[0]
    if isinstance(t, list):          # 旧式下拉 [["a","b"], {...}]
        return True
    if not isinstance(t, str):
        return False
    return t.strip("*") not in LINK_TYPES


def combo_default(spec):
    """下拉框缺值时取第一个可选项"""
    cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    d = cfg.get("default")
    if d is not None:
        return d
    opts = cfg.get("options")
    if isinstance(opts, list) and opts:
        o = opts[0]
        return o.get("key") if isinstance(o, dict) else o
    return None


def build_prompt(wf, oi):
    """前端 JSON（nodes/links）-> API prompt（{id: {class_type, inputs}}）"""
    links = {}
    for l in wf.get("links", []):
        # [link_id, 源节点, 源槽位, 目标节点, 目标槽位, 类型]
        links[l[0]] = (str(l[1]), l[2])
    prompt = {}
    for n in wf["nodes"]:
        ctype = n.get("type")
        if ctype in SKIP or ctype not in oi:
            continue
        info = oi[ctype]["input"]
        order = dict(list(info.get("required", {}).items()) +
                     list(info.get("optional", {}).items()))
        wv = list(n.get("widgets_values") or [])
        inputs, wi = {}, 0
        # 前端会在 seed 后面多存一个纯界面 widget（control_after_generate），
        # object_info 里没有它，按顺序硬对会整体错位一格
        wnames = [k for k, spec in order.items() if is_widget(spec)]
        if len(wv) == len(wnames) + 1 and "seed" in wnames:
            wv.pop(wnames.index("seed") + 1)
        for name in wnames:
            spec = order[name]
            if wi < len(wv):
                inputs[name] = wv[wi]
            else:
                # 工作流缺这个值（版本差异），退回节点默认值，避免缺参报错
                if isinstance(spec[0], str) and "COMBO" in spec[0]:
                    inputs[name] = combo_default(spec)
                else:
                    c = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                    inputs[name] = c.get("default")
            wi += 1
        for inp in n.get("inputs") or []:
            lid = inp.get("link")
            if lid is not None and lid in links:
                inputs[inp["name"]] = list(links[lid])
        prompt[str(n["id"])] = {"class_type": ctype, "inputs": inputs}
    return prompt


def build_payload(wf, oi, text, ref_names, w, h, length, steps, seed, prefix,
                  lora=None, lora_strength=1.0, vshift=12.0, ashift=6.0,
                  first_frame=None, last_frame=None, latent_upscaler=None):
    """在基础 prompt 上改条件节点：文生 / 首帧接力 / 多参考图，再挂 LoRA。

    参考图和首尾帧互斥（同一节点两种模式），同时给会被忽略。
    """
    p = build_prompt(wf, oi)
    n6 = p["6"]["inputs"]
    base = {"clip": n6["clip"], "vae": n6["vae"], "prompt": text,
            "width": w, "height": h, "length": length}

    if first_frame or last_frame:
        new6 = {"class_type": "MiniMaxH3ImageToVideo", "inputs": dict(base)}
    else:
        new6 = {"class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": dict(base, ref_image_size="match")}

    if first_frame or last_frame:
        if first_frame:
            p["210"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
            new6["inputs"]["first_frame"] = ["210", 0]
        if last_frame:
            p["211"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
            new6["inputs"]["last_frame"] = ["211", 0]
    else:
        for i, name in enumerate(ref_names or []):
            nid = str(100 + i)
            p[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
            # Autogrow 在 API 里展开为 ref_images.ref_image_N
            new6["inputs"]["ref_images.ref_image_%d" % i] = [nid, 0]
    p["6"] = new6

    # LoRA 必须插在 UNETLoader(2) 和 SigmaShift(3) 之间。
    # 放 Shift 之后，shift 会被 LoRA 权重覆盖，画面发灰发硬；
    # 同时 shift 要降到 turbo 档的 6/3，否则低步数采样欠火候。
    model_src = ["2", 0]
    if lora:
        p["200"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": model_src, "lora_name": lora,
                               "strength_model": float(lora_strength)}}
        model_src = ["200", 0]
    p["3"]["inputs"]["model"] = model_src
    p["3"]["inputs"]["shift_video"] = float(vshift)
    p["3"]["inputs"]["shift_audio"] = float(ashift)

    # resident 会把 4.6GB 文本编码器钉在显存里，采样阶段 8GB 卡必爆
    p["1"]["inputs"]["mode"] = "streaming"
    # 提示词重写预览分支出片用不到，而且它的 precision 枚举随节点版本变过
    #（"randomize" 现在非法），留着会让整张图校验失败
    p.pop("20", None)
    p.pop("21", None)

    p["9"]["inputs"]["steps"] = int(steps)
    p["10"]["inputs"]["noise_seed"] = int(seed)
    p["15"]["inputs"]["filename_prefix"] = prefix
    return p


# ---------------------------------------------------------------- 提交与轮询

def submit(payload):
    r = http_json(COMFY + "/prompt", {"prompt": payload})
    pid = r.get("prompt_id")
    if not pid:
        raise RuntimeError("ComfyUI 未返回 prompt_id: %s" % json.dumps(r, ensure_ascii=False)[:400])
    return pid, r


def wait(pid, timeout_s=7200, interval=6):
    """轮询到出结果。返回 (status, outputs, 耗时秒)"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(interval)
        try:
            h = http_json(COMFY + "/history/" + pid)
        except Exception:
            continue                  # ComfyUI 忙的时候 history 会短暂 500
        if pid in h:
            item = h[pid]
            return item.get("status", {}), item.get("outputs", {}), time.time() - t0
    return {"status_str": "timeout"}, {}, time.time() - t0


def err_text(status):
    for m in status.get("messages", []):
        if m[0] == "execution_error":
            return (m[1].get("exception_message", "") + " | node "
                    + str(m[1].get("node_id")))
    return json.dumps(status, ensure_ascii=False)[:500]


def oom(status):
    return "out of memory" in err_text(status).lower()


def find_video(outputs):
    """SaveVideo 的输出挂在 images 列表里（历史遗留字段名），按扩展名筛。"""
    for _nid, o in outputs.items():
        for item in o.get("images", []) or []:
            if str(item.get("filename", "")).lower().endswith(".mp4"):
                return item
    return None


def video_path(item):
    return os.path.join(COMFY_OUTPUT, item.get("subfolder", "") or "", item["filename"])


def free_vram():
    """让 ComfyUI 卸模型、清显存。换档位 / 换分辨率前调一次能少一半 OOM。"""
    try:
        http_json(COMFY + "/free", {"unload_models": True, "free_memory": True}, timeout=30)
        return True
    except Exception:
        return False


def comfy_alive():
    try:
        http_json(COMFY + "/system_stats", timeout=4)
        return True
    except Exception:
        return False


def ensure_ready():
    """启动前自检：给的是人话，不是堆栈。"""
    if not comfy_alive():
        raise SystemExit(
            "[ComfyUI 未连上] %s\n"
            "  1) 先启动 ComfyUI（默认 8188 端口）\n"
            "  2) 或改 config.json 里的 comfy_url" % COMFY)
    if not COMFY_DIR:
        raise SystemExit(
            "[缺 comfy_dir] config.json 里的 comfy_dir 要填 ComfyUI 根目录，\n"
            "  也就是能看到 input/ 和 output/ 两个文件夹的那一层。\n"
            "  例：\"comfy_dir\": \"D:/ComfyUI\"")
    for d in (COMFY_INPUT, COMFY_OUTPUT):
        if not os.path.isdir(d):
            raise SystemExit("[路径不对] 不存在: %s\n  检查 config.json 的 comfy_dir" % d)
    if not os.path.exists(WORKFLOW):
        raise SystemExit("[缺工作流] %s\n  检查 config.json 的 workflow" % WORKFLOW)


def stage_file(path):
    """把本地文件拷进 ComfyUI input，返回 LoadImage 用的文件名。

    相对路径先按仓库根解析，再按 cwd 兜底——从别的目录调用也不会缺文件。
    """
    cand = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.exists(cand):
        cand = os.path.abspath(path)
    if not os.path.exists(cand):
        raise SystemExit("[缺文件] %s" % path)
    name = os.path.basename(cand)
    dst = os.path.join(COMFY_INPUT, name)
    if os.path.abspath(cand) != os.path.abspath(dst):
        shutil.copyfile(cand, dst)
    return name


def load_workflow():
    with open(WORKFLOW, encoding="utf-8") as f:
        return json.load(f)


def object_info(timeout=120):
    return http_json(COMFY + "/object_info", timeout=timeout)


if __name__ == "__main__":
    print("comfy_url :", COMFY)
    print("comfy_dir :", COMFY_DIR or "(未配置)")
    print("workflow  :", WORKFLOW)
    print("output    :", OUTPUT_DIR)
    print("alive     :", comfy_alive())
    print("token 1280x704x124 =", tokens(1280, 704, 124), "/ limit", TOKEN_LIMIT)
