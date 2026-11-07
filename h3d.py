# -*- coding: utf-8 -*-
"""h3-director 引擎：H3 无限时长无缝出片（first_frame 接力 + 多参考图 + 音频 + 进化）

用法:
  python h3d.py plan --script story.txt          # 预览分段计划
  python h3d.py run  --script story.txt [--name 任务名] [--resume]
  python h3d.py stats                            # 查看进化状态（档位成功率/耗时）

分镜脚本格式见同目录 SKILL.md。
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h3core as H3            # noqa: E402

STATE = os.path.join(HERE, "state")
RUNS = os.path.join(STATE, "runs.jsonl")
PARAMS = os.path.join(STATE, "params.json")
LESSONS = os.path.join(STATE, "lessons.md")

OUT_ROOT = os.path.join(H3.OUTPUT_DIR, "h3d")
COMFY_INPUT = H3.COMFY_INPUT or ""
FFMPEG = H3.CFG.get("ffmpeg") or "ffmpeg"
FFPROBE = H3.CFG.get("ffprobe") or "ffprobe"

FPS = 24
FRAMES = 124                    # 每段固定 5.167 秒：124 帧是三档都零崩的帧数
SEG_DUR = FRAMES / FPS
TARGET_W, TARGET_H = 1280, 720  # 交付分辨率：拼接时统一 lanczos 放大
XFADE = 3                       # 接缝交叉溶解帧数
AFD = XFADE / FPS               # 音频交叉时长（秒）

# 三档（梁总命名：低 / 中 / 高，默认中档）。英文别名兼容命令行传参
TIERS = {
    "低": dict(w=704, h=384, steps=4, lora=H3.LORA_NAME,
              vshift=6.0, ashift=3.0, sharpen=True),
    "中": dict(w=768, h=432, steps=8, lora="",
              vshift=12.0, ashift=6.0, sharpen=False),
    # 高档 = 三采：768x432 起手 → latent 放大 1152x640 精修 → 1280x704 再精修
    # token 账：48k → 106k → 130k（130k 逼近 8GB 硬限 133k，慢但不出崩）
    "高": dict(w=768, h=432, steps=8, lora="",
              vshift=12.0, ashift=6.0, sharpen=True,
              passes=[dict(w=1152, h=640, steps=4, denoise=0.55),
                      dict(w=1280, h=704, steps=3, denoise=0.30)]),
}
TIER_ALIAS = {"fast": "低", "low": "低", "stable": "中", "mid": "中", "medium": "中",
              "hq": "高", "high": "高",
              "低": "低", "中": "中", "高": "高"}
DEFAULT_TIER = "中"


def norm_tier(v):
    """英文别名/旧值 → 中文档位名。命令行传中文怕乱码，别名为准。"""
    return TIER_ALIAS.get(str(v).strip(), str(v).strip())


def apply_passes(p, passes, seed, upscaler=None):
    """在主管道后串多级 latent 放大 + 低 denoise 精修（二采/三采通用）。

    H3 的 latent 是音视频合并的 NestedTensor，只能用 MiniMaxH3LatentUpscaleWithModel
    （音频原样透传）。每级 = 放大 → 新 scheduler/noise/sampler，接 12/13 的 samples。
    """
    loader = "219"
    if loader not in p:
        p[loader] = {"class_type": "LatentUpscaleModelLoader",
                     "inputs": {"model_name": upscaler or H3.LATENT_UPSCALER}}
    src = "11"                       # 主采样 SamplerCustomAdvanced 的输出
    for i, ps in enumerate(passes):
        b = 320 + i * 6
        n = lambda k: str(b + k)     # noqa: E731
        p[n(0)] = {"class_type": "MiniMaxH3LatentUpscaleWithModel",
                   "inputs": {"model": [loader, 0], "samples": [src, 0],
                              "width": int(ps["w"]), "height": int(ps["h"])}}
        p[n(1)] = {"class_type": "BasicScheduler",
                   "inputs": {"scheduler": "simple", "steps": int(ps["steps"]),
                              "denoise": float(ps["denoise"]), "model": ["3", 0]}}
        p[n(2)] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed) + i + 1}}
        p[n(3)] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": [n(2), 0], "guider": ["7", 0], "sampler": ["8", 0],
                              "sigmas": [n(1), 0], "latent_image": [n(0), 0]}}
        src = n(3)
        print(f"  [第{i + 2}采] {ps['w']}x{ps['h']} / {ps['steps']} 步 denoise {ps['denoise']}")
    p["12"]["inputs"]["samples"] = [src, 0]
    p["13"]["inputs"]["samples"] = [src, 0]


# ============================ 进化层 ============================

def _load_params():
    if os.path.exists(PARAMS):
        return json.load(open(PARAMS, encoding="utf-8"))
    return {"tiers": {}}


def bump_stats(tier, ok, secs, err=None, seam=None):
    """每次段生成后更新统计。连续失败会被 run 启动检查拦截。"""
    p = _load_params()
    t = p["tiers"].setdefault(tier, {"ok": 0, "fail": 0, "streak_fail": 0,
                                     "avg_secs": 0.0, "last_err": "", "seams": []})
    if ok:
        t["ok"] += 1
        t["streak_fail"] = 0
        n = t["ok"]
        t["avg_secs"] = round((t["avg_secs"] * (n - 1) + secs) / n, 1)
        if seam is not None:
            t["seams"] = (t["seams"] + [round(seam, 4)])[-20:]
    else:
        t["fail"] += 1
        t["streak_fail"] += 1
        t["last_err"] = (err or "")[:200]
    json.dump(p, open(PARAMS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def note_seam(tier, s):
    """只追加接缝分数，不动成功/失败计数（那段已经计过数了）。"""
    if s is None:
        return
    p = _load_params()
    t = p["tiers"].setdefault(tier, {"ok": 0, "fail": 0, "streak_fail": 0,
                                     "avg_secs": 0.0, "last_err": "", "seams": []})
    t["seams"] = (t.get("seams") or [] + [round(float(s), 4)])[-20:]
    json.dump(p, open(PARAMS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def check_tier(tier):
    """启动自检：某档连续失败 >=2 就拦下来，逼着换档而不是撞墙。"""
    t = _load_params()["tiers"].get(tier)
    if t and t.get("streak_fail", 0) >= 2:
        raise SystemExit(
            f"[进化拦截] 档位 {tier} 已连续失败 {t['streak_fail']} 次，"
            f"最近错误: {t.get('last_err')}\n"
            f"换档位（--tier 另一档）或先看 stats / lessons.md 再来。")


def record(ev):
    os.makedirs(STATE, exist_ok=True)
    ev["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


# ============================ 分镜解析 ============================

META_KEYS = ("tier", "refs", "audio", "out", "negative", "seed", "cfg", "wide", "res")


def parse_story(path):
    meta = {"tier": DEFAULT_TIER, "refs": [], "audio": None, "out": None,
            "negative": None, "seed": None, "cfg": None}
    segs = []
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            k, _, v = line[1:].partition(" ")
            k = k.strip().lower()
            if k not in META_KEYS:
                raise SystemExit(f"[分镜] 不认识的指令 @{k}（可用: {META_KEYS}）")
            if k == "refs":
                meta[k] = [x for x in v.split() if x]
            elif k == "seed":
                meta[k] = int(v)
            elif k == "tier":
                meta[k] = norm_tier(v)
            elif k == "cfg" or k == "wide":
                meta[k] = float(v)
            else:
                meta[k] = v.strip() or None
        elif line == "---":
            continue
        else:
            segs.append(line)
    if not segs:
        raise SystemExit("[分镜] 一段都没有")
    if meta["tier"] not in TIERS:
        raise SystemExit(f"[分镜] tier 只能是 {list(TIERS)}（别名 {list(TIER_ALIAS)}）")
    return meta, segs


def show_plan(meta, segs):
    t = TIERS[meta["tier"]]
    print(f"任务: {meta['out'] or '(未命名)'}  档位 {meta['tier']} "
          f"({t['w']}x{t['h']} / {t['steps']} 步 / lora={bool(t['lora'])})")
    print(f"参考图: {meta['refs'] or '无'}  音频: {meta['audio'] or '无'}")
    print(f"段数 {len(segs)} x 5.17s ≈ 总时长 {len(segs) * SEG_DUR - (len(segs)-1) * XFADE / FPS:.1f}s")
    for i, s in enumerate(segs):
        tag = "ref起手" if i == 0 and meta["refs"] else ("接力" if i else "t2v")
        print(f"  seg{i:02d} [{tag}] {s[:60]}{'…' if len(s) > 60 else ''}")


# ============================ 段生成 ============================

def stage_into_input(path):
    """把本地文件拷进 ComfyUI input，返回 LoadImage 用的文件名（实现在 h3core）。"""
    return H3.stage_file(path)


def grab_frame(mp4, dst, w, h, seek_end=True):
    """抽首帧(False)/末帧(True)并缩回生成分辨率——接力帧必须和下段生成尺寸一致。"""
    cmd = [FFMPEG, "-y", "-v", "error"]
    if seek_end:
        cmd += ["-sseof", "-0.15", "-i", mp4]
    else:
        cmd += ["-i", mp4]
    cmd += ["-frames:v", "1", "-vf", f"scale={w}:{h}:flags=lanczos", dst]
    subprocess.run(cmd, check=True)
    return dst


def probe_has_audio(mp4):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=codec_type", "-of", "csv=p=0", mp4],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


SEAM_SCALE = 160              # 结构判定用的缩略边长
SEAM_WARN = 0.90              # 结构层 SSIM 低于此值才判为漂移


def _ssim(a, b, size=None, tmpdir=None):
    """ffmpeg 算 SSIM。size 给定时先把两张图缩到 size×size（area 采样）。"""
    if size:
        a2 = os.path.join(tmpdir, "_sa.png")
        b2 = os.path.join(tmpdir, "_sb.png")
        for src, dst in ((a, a2), (b, b2)):
            subprocess.run([FFMPEG, "-y", "-v", "error", "-i", src,
                            "-vf", f"scale={size}:{size}:flags=area", dst], check=True)
        a, b = a2, b2
    r = subprocess.run(
        [FFMPEG, "-v", "info", "-i", a, "-i", b,
         "-filter_complex", "ssim", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"All:([0-9.]+)", r.stderr)
    return float(m.group(1)) if m else None


def seam_ssim(prev_png, next_png, tmpdir=None):
    """接缝量化，两级指标。

    full   —— 原分辨率 SSIM。对纹理、雾气、树叶、瓦片这类高频细节极敏感，
              高质量画面反而分数低，只能当参考。
    struct —— 缩到 160px 后再算，滤掉高频噪声，只留构图与色调，**用它判定**。

    实测（同一批接力，肉眼全部连贯）：
       静谧场景   full 0.87~0.90  struct 0.96~0.97
       人物中景   full 0.90       struct 0.96
       瓦片+雾    full 0.74~0.78  struct 0.92~0.94   ← 全分辨率会误报
    所以警戒线放在 struct 0.90，而不是 full 0.85。
    """
    tmpdir = tmpdir or tempfile.gettempdir()
    full = _ssim(prev_png, next_png)
    struct = _ssim(prev_png, next_png, SEAM_SCALE, tmpdir)
    return {"full": full, "struct": struct}


def gen_seg(i, text, meta, oi, job_dir, ff_name=None, tier=None):
    """生成第 i 段。ff_name=ComfyUI input 里的接力帧名；首段带 refs。"""
    tier = tier or meta["tier"]
    t = TIERS[tier]
    seed = (meta["seed"] if meta["seed"] is not None else random.randrange(1 << 30)) + i
    prefix = f"h3d_{meta['out'] or 'job'}_seg{i:02d}"
    refs = [stage_into_input(x) for x in meta["refs"]] if (i == 0 and meta["refs"]) else []

    wf = H3.load_workflow()
    p = H3.build_payload(wf, oi, text, refs, t["w"], t["h"], FRAMES, t["steps"],
                        seed, prefix, lora=t["lora"], vshift=t["vshift"], ashift=t["ashift"],
                        first_frame=ff_name if i > 0 else None)
    # 无 LoRA 分支 build_payload 不写 shift，这里统一写死，档位语义才成立
    p["3"]["inputs"]["shift_video"] = float(t["vshift"])
    p["3"]["inputs"]["shift_audio"] = float(t["ashift"])

    if meta["negative"]:
        # H3 原生 BasicGuider 无负面条件，换 CFGGuider 挂负面（蒸馏模型 cfg 1.5 不发硬）
        p["300"] = {"class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["1", 0], "text": meta["negative"]}}
        p["7"] = {"class_type": "CFGGuider",
                  "inputs": {"model": p["7"]["inputs"]["model"],
                             "positive": p["7"]["inputs"]["conditioning"],
                             "negative": ["300", 0],
                             "cfg": float(meta["cfg"] if meta["cfg"] is not None else 1.5)}}

    if t.get("passes"):
        apply_passes(p, t["passes"], seed)

    pid, _ = H3.submit(p)
    st, outs, secs = H3.wait(pid, timeout_s=7200)
    if st.get("status_str") != "success":
        err = H3.err_text(st)
        bump_stats(tier, False, secs, err=err)
        record({"kind": "seg", "i": i, "tier": tier, "ok": False, "secs": round(secs), "err": err})
        raise RuntimeError(f"seg{i} 失败: {err}")
    item = H3.find_video(outs)
    if not item:
        raise RuntimeError(f"seg{i} 成功但找不到 mp4 输出")
    src = os.path.join(H3.COMFY_OUTPUT, item.get("subfolder", ""), item["filename"])
    dst = os.path.join(job_dir, f"seg{i:02d}.mp4")
    shutil.copyfile(src, dst)
    bump_stats(tier, True, secs)
    record({"kind": "seg", "i": i, "tier": tier, "ok": True, "secs": round(secs, 1)})
    print(f"  seg{i:02d} 完成 {secs / 60:.1f} 分 -> {os.path.basename(dst)}")
    return dst


def run_seg_with_recovery(i, text, meta, oi, job_dir, ff_src=None):
    """带失败恢复：OOM 自动 stable->fast 降档重试一次；随机错误 seed 换一次。"""
    ff_name = None
    if ff_src:
        ff_name = f"h3d_ff_{os.path.basename(job_dir)}_{i:02d}.png"
        shutil.copyfile(ff_src, os.path.join(COMFY_INPUT, ff_name))
    tier = meta["tier"]
    try:
        return gen_seg(i, text, meta, oi, job_dir, ff_name, tier)
    except RuntimeError as e:
        if H3.oom({"messages": [["execution_error",
                                 {"exception_message": str(e), "node_id": "?"}]]}):
            if tier == "中":
                print(f"  [降档] seg{i} OOM，中 -> 低 重试")
                return gen_seg(i, text, meta, oi, job_dir, ff_name, "低")
            raise
        print(f"  [重试] seg{i} 一次: {e}")
        base = meta["seed"] if meta["seed"] is not None else random.randrange(1 << 30)
        meta2 = dict(meta, seed=base + random.randrange(1000, 9999))
        return gen_seg(i, text, meta2, oi, job_dir, ff_name, tier)


# ============================ 拼接与音频 ============================

def join_video(parts, dst, sharpen=False):
    """lanczos 统一放大到 1280x720 + 3 帧 xfade 接力。sharp 档另加轻度锐化。"""
    n = len(parts)
    fc = []
    for i in range(n):
        shp = ",unsharp=5:5:0.5" if sharpen else ""
        fc.append(f"[{i}:v]scale={TARGET_W}:{TARGET_H}:flags=lanczos,setsar=1{shp}[v{i}]")
    prev, offset = "v0", 0.0
    for i in range(1, n):
        offset += SEG_DUR - XFADE / FPS
        out = f"x{i}"
        fc.append(f"[{prev}][v{i}]xfade=transition=fade:duration={XFADE / FPS:.4f}"
                  f":offset={offset:.4f}[{out}]")
        prev = out
    cmd = [FFMPEG, "-y", "-v", "error"]
    for p in parts:
        cmd += ["-i", p]
    cmd += ["-filter_complex", ";".join(fc), "-map", f"[{prev}]",
            "-r", str(FPS), "-c:v", "libx264", "-crf", "17", "-preset", "medium", dst]
    subprocess.run(cmd, check=True)


def join_audio(parts, dst):
    """H3 段原声 acrossfade 链 + 降噪 -18dB。任何一段没音轨就放弃段声版。"""
    if not all(probe_has_audio(p) for p in parts):
        return False
    n = len(parts)
    fc, prev = [], "0:a"
    for i in range(1, n):
        out = f"a{i}"
        fc.append(f"[{prev}][{i}:a]acrossfade=d={AFD:.4f}[{out}]")
        prev = out
    fc.append(f"[{prev}]afftdn=nr=12,loudnorm=I=-18:TP=-1.5:LRA=11[aout]")
    cmd = [FFMPEG, "-y", "-v", "error"]
    for p in parts:
        cmd += ["-i", p]
    cmd += ["-filter_complex", ";".join(fc), "-map", "[aout]", "-c:a", "aac", "-b:a", "160k", dst]
    subprocess.run(cmd, check=True)
    return True


def _upscale_out(mp4, res):
    """交付分辨率放大。8GB 卡原生只能到 1280x704（token 硬限），
    1080p 走 lanczos 2x + 轻锐化——是「交付级 1080p」，细节靠补不靠造。"""
    if not res:
        return
    targets = {"1080p": (1920, 1080), "1440p": (2560, 1440), "2k": (2560, 1440), "4k": (3840, 2160)}
    key = str(res).lower().rstrip("p") + "p" if str(res).lower().rstrip("p").isdigit() else str(res).lower()
    tw, th = targets.get(key, (0, 0))
    if not tw:
        print(f"[分辨率] 不认识的 @res {res}，可选 {list(targets)}")
        return
    up = mp4 + ".up.mp4"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", mp4,
                    "-vf", f"scale={tw}:{th}:flags=lanczos,unsharp=5:5:0.4:5:5:0.0",
                    "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                    "-c:a", "copy", up], check=True)
    shutil.move(up, mp4)
    print(f"[分辨率] 交付放大到 {tw}x{th}（注意：细节是补的，非原生生成）")


def mux_bgm(video, bgm, dst, total):
    cmd = [FFMPEG, "-y", "-v", "error", "-i", video,
           "-stream_loop", "-1", "-i", bgm,
           "-t", f"{total:.3f}", "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", dst]
    subprocess.run(cmd, check=True)


# ============================ 主流程 ============================

def cmd_run(a):
    meta, segs = parse_story(a.script)
    if a.tier:
        meta["tier"] = norm_tier(a.tier)
        if meta["tier"] not in TIERS:
            raise SystemExit(f"[档位] 只能是 低/中/高（别名 fast/stable/hq）")
    if a.refs:
        meta["refs"] = a.refs
    if a.audio:
        meta["audio"] = a.audio
    if a.res:
        meta["res"] = a.res
    check_tier(meta["tier"])

    name = a.name or meta["out"] or time.strftime("h3d_%m%d_%H%M")
    job = os.path.join(OUT_ROOT, name)
    os.makedirs(job, exist_ok=True)
    seam_dir = os.path.join(job, "seam_check")
    os.makedirs(seam_dir, exist_ok=True)
    manifest_p = os.path.join(job, "manifest.json")
    mf = {"name": name, "meta": meta, "segs": []}
    if a.resume and os.path.exists(manifest_p):
        mf = json.load(open(manifest_p, encoding="utf-8"))
        print(f"[续跑] 已完成 {sum(1 for s in mf['segs'] if s['status'] == 'done')}/{len(segs)} 段")
    mf.setdefault("segs", [dict(status="none") for _ in segs])
    while len(mf["segs"]) < len(segs):
        mf["segs"].append(dict(status="none"))

    show_plan(meta, segs)
    if a.dry:
        return

    H3.ensure_ready()                 # ComfyUI 在不在、路径对不对，先给人话
    oi = H3.object_info()
    H3.free_vram()                    # 清一次显存，换档位后 OOM 概率减半
    t0 = time.time()
    for i, text in enumerate(segs):
        st = mf["segs"][i]
        if st.get("status") == "done" and os.path.exists(st.get("file", "")):
            print(f"  seg{i:02d} 已完成，跳过")
            continue
        # 接力帧：上段 mp4 的末帧缩回生成分辨率
        ff_src = None
        if i > 0:
            prev = mf["segs"][i - 1].get("file")
            if not prev or not os.path.exists(prev):
                raise SystemExit(f"[接力断档] seg{i-1} 的成片不存在，先补跑它")
            tt = TIERS[mf["segs"][i - 1].get("tier") or meta["tier"]]
            ff_src = os.path.join(job, f"_ff_{i - 1:02d}.png")
            grab_frame(prev, ff_src, tt["w"], tt["h"], seek_end=True)
        mp4 = run_seg_with_recovery(i, text, meta, oi, job, ff_src)
        st.update(status="done", file=mp4, tier=meta["tier"])
        # 接缝校验：下段首帧(生成结果) vs 上段末帧(接力输入)
        if i > 0 and ff_src:
            head = os.path.join(seam_dir, f"seg{i:02d}_head.png")
            grab_frame(mp4, head, TARGET_W, TARGET_H, seek_end=False)
            prev_end = os.path.join(seam_dir, f"seg{i - 1:02d}_tail.png")
            grab_frame(mf["segs"][i - 1]["file"], prev_end, TARGET_W, TARGET_H, seek_end=True)
            sm = seam_ssim(prev_end, head, seam_dir)
            st["seam"] = sm["struct"]          # 结构层分数：判定用
            st["seam_full"] = sm["full"]       # 逐像素分数：仅参考
            if sm["struct"] is not None:
                print(f"  [接缝] seg{i} struct={sm['struct']:.3f} full={sm['full']:.3f}")
            note_seam(meta["tier"], sm["struct"])
            if sm["struct"] is not None and sm["struct"] < SEAM_WARN:
                print(f"  [警告] seg{i} 接缝 struct={sm['struct']:.3f} < {SEAM_WARN}，"
                      f"接力漂移，看 seam_check/")
            json.dump(mf, open(manifest_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 拼接 ----
    print("[拼接] 放大到 1280x720 + xfade…")
    parts = [mf["segs"][i]["file"] for i in range(len(segs))]
    total = len(parts) * SEG_DUR - (len(parts) - 1) * XFADE / FPS
    joined = os.path.join(job, "_joined.mp4")
    join_video(parts, joined, sharpen=TIERS[meta["tier"]]["sharpen"])
    main_out = os.path.join(job, f"{name}.mp4")
    voice = os.path.join(job, "_h3audio.m4a")
    has_voice = join_audio(parts, voice)
    bgm = meta["audio"]
    if bgm and not (os.path.isabs(bgm) and os.path.exists(bgm)):
        bgm = stage_into_input(bgm)
    if bgm and has_voice:
        # 配乐铺底压低 9dB + 段原声（台词/环境音）混音，两者都保留
        fc = (f"[2:a]atrim=0:{total:.3f},volume=-9dB[bg];"
              f"[1:a][bg]amix=inputs=2:duration=first:normalize=0[aout]")
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", joined, "-i", voice,
                        "-stream_loop", "-1", "-i", bgm, "-filter_complex", fc,
                        "-map", "0:v", "-map", "[aout]", "-shortest",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", main_out], check=True)
        print("[音频] 段原声 + 配乐混音版（主交付）")
    elif bgm:
        mux_bgm(joined, bgm, main_out, total)
        print("[音频] 段无声轨，纯配乐版（主交付）")
    elif has_voice:
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", joined, "-i", voice,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-shortest", main_out], check=True)
        print("[音频] 段原声 loudnorm -18 LUFS 版（主交付）")
    else:
        shutil.copyfile(joined, main_out)
        print("[警告] 段无音轨且没给 @audio，只能出静音版——下次在 prompt 里写环境音")
    os.remove(joined)
    # 宽幅裁切：2.35:1 裁掉底部字幕条（参考图带剧集先验时会出字，裁是唯一根治）
    wide = meta.get("wide")
    if wide and wide >= 1.5:
        ch = int(TARGET_W / wide) // 2 * 2
        cy = (TARGET_H - ch) // 2 // 2 * 2
        cut = os.path.join(job, "_wide.mp4")
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", main_out,
                        "-vf", f"crop={TARGET_W}:{ch}:0:{cy}", "-c:a", "copy", cut], check=True)
        shutil.move(cut, main_out)
        print(f"[宽幅] 裁到 {TARGET_W}x{ch} (2.35:1)，底部字幕条已去除")
    _upscale_out(main_out, meta.get("res"))
    record({"kind": "job", "ok": True, "name": name, "tier": meta["tier"], "segs": len(segs),
            "total": round(total, 1), "secs": round(time.time() - t0)})
    json.dump(mf, open(manifest_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n完成: {main_out}  ({len(segs)} 段 / {total:.1f}s / 总耗时 {(time.time()-t0)/60:.1f} 分)")
    print(f"经验沉淀：把本轮新踩的坑追加到 {LESSONS}")


def cmd_stats(_):
    p = _load_params()
    print(f"{'档位':<8}{'成功':>5}{'失败':>5}{'连败':>5}{'均耗(分)':>9}  最近错误")
    for k, t in p["tiers"].items():
        print(f"{k:<8}{t['ok']:>5}{t['fail']:>5}{t.get('streak_fail', 0):>5}"
              f"{t.get('avg_secs', 0) / 60:>9.1f}  {t.get('last_err', '')[:60]}")
    seams = [s for t in p["tiers"].values() for s in t.get("seams", [])]
    if seams:
        print(f"\n接缝 struct: 最近均值 {sum(seams) / len(seams):.3f} "
              f"(最低 {min(seams):.3f}，警戒线 {SEAM_WARN})")
    if os.path.exists(RUNS):
        print("\n最近运行:")
        with open(RUNS, encoding="utf-8") as f:
            for ln in f.readlines()[-8:]:
                e = json.loads(ln)
                print(" ", e["ts"], e["kind"], e.get("name") or f"seg{e.get('i')}",
                      norm_tier(e.get("tier", "")), "ok" if e.get("ok") else "FAIL",
                      f"{e.get('secs', '?')}s", (e.get("err") or "")[:50])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="出片")
    pr.add_argument("--script", required=True)
    pr.add_argument("--name")
    pr.add_argument("--tier", help="档位：低/中/高（或 fast/stable/hq）")
    pr.add_argument("--refs", nargs="*")
    pr.add_argument("--audio")
    pr.add_argument("--resume", action="store_true")
    pr.add_argument("--res", help="交付分辨率放大：1080p / 1440p / 4k")
    pr.add_argument("--dry", action="store_true", help="只打印计划")
    pr.set_defaults(fn=cmd_run)
    pp = sub.add_parser("plan", help="预览分段计划")
    pp.add_argument("--script", required=True)
    pp.set_defaults(fn=lambda a: show_plan(*parse_story(a.script)))
    ps = sub.add_parser("stats", help="进化状态")
    ps.set_defaults(fn=cmd_stats)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
