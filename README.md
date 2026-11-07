# h3-director

**MiniMax H3 无限时长无缝出片导演台。** 基于 ComfyUI，把一个分镜脚本拆成任意多段，逐段用上一段末帧做 `first_frame` 接力，拼成一条看不出接缝的长片。

带一个浏览器界面，不会命令行也能点着用。

```
分镜脚本（每行 = 5.17 秒）
        ↓
  段0 (文生/参考图)  →  段1 (首帧接力)  →  段2 (首帧接力)  →  ...
        ↓                    ↓                  ↓
      末帧 ──────────────→ 首帧 ─────────────→ 首帧
        ↓
  xfade 3 帧拼接 + 段原声 acrossfade + 交付放大
```

## 它解决什么

单次生成只能出 5 秒左右。想要 30 秒、1 分钟、3 分钟的一镜到底，常见的做法是各段独立生成再硬拼 —— 结果人物换脸、光线跳变、动作断裂，一眼就看出是拼的。

这里的做法是**帧级接力**：每段生成完后抽末帧，作为下一段的 `first_frame`，模型从这一帧继续往下画。加上 3 帧交叉溶解，接缝处结构层 SSIM 0.92–0.97（肉眼不可分辨）。

## 三个档位

档位名就是 **低 / 中 / 高**，英文别名 `fast` / `stable` / `hq`。默认中档。

| 档位 | 生成分辨率 | 采样 | 实测耗时/段 | 适用 |
|---|---|---|---|---|
| **低** | 704×384 | 4 步 + turbo LoRA | ≈1 分 50 秒 | 无对白的空镜、环境镜、打斗过渡 |
| **中**（默认） | 768×432 | 8 步全采样 | ≈3 分 | 有对白、要人声、怕字幕 —— 一律用这个 |
| **高** | 768×432 → 1152×640 → 1280×704 | 三采（8 + 4 + 3 步） | ≈11 分 | 特写、收藏级画质 |

全部按交付 1280×720 / 24fps 输出。`@res 1080p` 可再放大到 1920×1080（后处理，非原生生成，见下文限制）。

**低档有个坑，写在前面**：4 步蒸馏 LoRA 会把音频分支一起蒸坏，台词会含混听不清；704×384 还容易落进训练分布里「低清 = 带字幕压缩剧集」的区域，画面底部冒出对白字幕。中档两项都没有。所以**只要这段有台词，就必须中档起**。

## 硬前提

1. **ComfyUI**（≥ 0.3.40，带 MiniMax H3 节点）跑在 `127.0.0.1:8188`
2. **ffmpeg / ffprobe** 在 PATH 里
3. **Python 3.9+**（只用标准库，不需要 pip install）
4. 模型文件（放 ComfyUI 对应目录）：

| 目录 | 文件 |
|---|---|
| `models/diffusion_models/` | `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` |
| `models/vae/` | `minimax_h3_video_vae_int8_convrot.safetensors`、`minimax_h3_audio_vae_fp32.safetensors` |
| `models/text_encoders/` | `qwen3vl_4b_fp8_scaled.safetensors` |
| `models/clip_proj/` | `mmh3-4b-ClipProj-v3.1-mlp.safetensors` |
| `models/loras/` | `pika_h3_flex_turbo_4step_fl2va_lite_r32.safetensors`（低档用，可不要） |
| `models/latent_upscale_models/` | `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`（高档用） |

工作流用的 ClipProj 方案（把 15.7GB 的 Qwen3-VL-32B 换成 4B + 投影矩阵）来自社区，下载链接写在工作流里的三个 MarkdownNote 节点上，ComfyUI 里打开就能看到。

5. **显存**：8GB 能跑，但要认命 —— 单次采样 token 上限约 133,200（1280×704×124 帧就是踩线）。16GB 以上可以把 `config.json` 里的模型换成 fp8/bf16 版本，分辨率上限随之上移。

## 装

```bash
git clone <this-repo> h3-director
cd h3-director
cp config.example.json config.json
```

改 `config.json` 里两行就能跑：

```json
{
  "comfy_dir": "D:/ComfyUI",              // 能看到 input/ 和 output/ 的那一层
  "comfy_url": "http://127.0.0.1:8188"
}
```

自检：

```bash
python h3core.py
# comfy_url : http://127.0.0.1:8188
# comfy_dir : D:/ComfyUI
# alive     : True
# token 1280x704x124 = 130240 / limit 133200
```

## 用

### 命令行

```bash
python h3d.py plan --script presets/story_example.txt    # 只看计划，不生成
python h3d.py run  --script presets/story_example.txt    # 出片
python h3d.py run  --script myshot.txt --resume          # 断了接着跑（段级断点）
python h3d.py stats                                      # 历史成功率、耗时、接缝分数
```

产物：`output/h3d/<任务名>/<任务名>.mp4`，附 `manifest.json`（每段状态、接缝 SSIM）和 `seam_check/`（接缝前后帧对比图）。

### 浏览器界面

```bash
python ui/h3d_server.py          # 打开 http://127.0.0.1:8190
```

Windows 双击 `启动导演台.bat` 也行。

界面上：选档位 → 拖参考图 → 写分镜（一格一段）→ 点生成 → 看日志 → 成片库里直接播放。

## 分镜脚本格式

```
# 井号开头是注释
@tier 中
@refs refs/face.png refs/scene.png
@audio bgm.mp3
@out 我的片子
@seed 42
---
第一段的画面描述。环境音：风声，远处狗叫。
第二段。
第三段。
```

`---` 之后，一行就是一段，段数不限。

| 指令 | 说明 |
|---|---|
| `@tier 低/中/高` | 档位，默认中。别名 `fast/stable/hq` 也认 |
| `@refs a.png b.png` | 参考图，最多 5 张（4 张脸 + 1 张场景比较合适），**只在第一段生效** |
| `@audio x.mp3` | 配乐，自动循环铺满，压低 9dB 与段原声混音 |
| `@out 名字` | 任务名，也是输出文件名 |
| `@seed 42` | 种子，留空则随机 |
| `@negative 字幕, 水印` | 负面提示（换成 CFGGuider，cfg 1.5） |
| `@wide 2.35` | 输出裁成 2.35:1 宽幅，用来裁掉底部字幕条 |
| `@res 1080p` | 交付放大到 1920×1080（可选 1440p / 4k） |

### 写分镜的三条经验

1. **要声音就得写声音。** H3 的音频分支按提示词生成音轨，提示词里不写环境音，出来的可能近乎全静。写「环境音：流水声，鸟鸣」这种就够了。
2. **要角色出声，必须连动作一起写。** 「猫叫」不会让猫张嘴；「抬头**张嘴**叫」才会。H3 的音频和画面是两个并行分支，没有对口型机制 —— 声画对齐只能后期做。
3. **参考图和首帧接力互斥。** 参考图只在第一段起手，之后全靠接力把人物和场景带下去。接力链会缓慢漂移，所以建议每 6–8 段让人物重新「首次出场」一次（重新用 ref 起手）来重锚。

## 已知边界

- **单段固定 124 帧**（5.17 秒）。改帧数要同步改 `h3d.py` 里的 `FRAMES`，且必须落在 H3 的时间网格上（17 帧 5 步的节奏），乱改会直接 OOM 或出 2 帧。
- **token 天花板**。8GB 卡单次采样 133,200 token 是硬限。1920×1080×124 帧需要 301,920，是它的 2.3 倍 —— 不是慢，是装不下。要原生 1080p 得上 24GB 卡。
- **`@res 1080p` 是补不是造**。它是 lanczos 2× + 轻锐化，细节靠插值撑，等效清晰度大概 1440p 档，比原生 1080p 少一层微观信息。
- **接缝分数分两级**。`struct`（缩到 160px 后算 SSIM）是判定用的，低于 **0.90** 才算漂移；`full`（原分辨率）对雾气、树叶、瓦片这类高频细节极敏感，高质量画面反而分数低，只作参考 —— 同一批肉眼连贯的接缝，`full` 能从 0.74 到 0.90 波动，别拿它当标准。`seam_check/` 里有前后帧对比图，肉眼比数字可靠。
- **跨段音色会漂**。每段独立生成音频，同一角色跨段音色不保证一致，台词还可能被切断。对策：音色描述逐段复写同一句；台词排在段内前 60% 说完，结尾留半拍静默。
- **界面无鉴权**。`--host 0.0.0.0` 会暴露给局域网，自己注意。

## 目录

```
h3-director/
├── h3d.py                 引擎：分镜解析、段生成、接力、拼接、音频、进化统计
├── h3core.py              ComfyUI 客户端：工作流转 API、token 预算、提交轮询
├── config.json            你的配置（不进 git）
├── config.example.json    配置模板
├── workflows/
│   └── h3_base_124f.json  基础工作流（21 节点，前端格式）
├── ui/
│   ├── h3d_server.py      可视化后端（标准库 HTTP）
│   └── index.html         单页前端
├── presets/
│   └── story_example.txt  示例分镜
├── state/                 运行时生成：runs.jsonl（历史）、params.json（档位统计）、lessons.md（经验）
└── docs/
    └── 调参笔记.md        实测数据与踩坑记录
```

## 进化机制

`h3d.py` 每跑一段都会记一条 `state/runs.jsonl`，并按档位累计成功率、平均耗时、接缝分数到 `state/params.json`。某个档位**连续失败 2 次**后，下次启动会被直接拦下并报出最近的错误 —— 逼你换档，而不是反复撞墙。

这不是 AI 自动学习，是把「哪档在什么机器上跑得动」量化下来，让下次决策有据可依。想重置就直接删 `state/params.json`。

## 许可

MIT。见 [LICENSE](LICENSE)。

基础工作流（`workflows/h3_base_124f.json`）里的 ClipProj 说明节点来自社区分享，保留原样以方便查到模型出处。
