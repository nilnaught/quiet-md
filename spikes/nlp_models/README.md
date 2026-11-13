# `spikes/nlp_models` — 离线神经词性 + 依存 spike（T-spike-1 / T-spike-2）

验证：**完全离线**的中英词性标注 + 依存分析能否用 **Rust + `ort`（ONNX
Runtime）在 CPU 上**跑通，并据此派生阅读标记（名词/动词/形容词/副词着色、
连词下划线、英文非谓语虚线、核心谓语着重号）。

中文用 **LTP/small**（ltp-core），英文用
**KoichiYasuoka/roberta-{large,base}-english-upos**（UPOS）+ 它在 `esupar.model`
里捆绑的 **SuPar biaffine 依存**。spaCy `en_core_web_trf` **只作对照**。

英文有 **4 个变体**（`--en-variant`）：`large-fp32`（默认，与旧路径兼容）、
`large-int8`、`base-fp32`、`base-int8`。比较与推荐见
`/workspace/spike-out/en_compare.md`；**默认推荐 `base-int8`**
（373 MiB / 0.5 s 加载 / ~26 ms/句 / 323 MiB RSS，粗 UPOS 98.1%，与 large 持平）。

```
spikes/nlp_models/
  README.md                  ← 本文件
  EXPORT.md                  ← 导出过程、结构审计、失败与坑（必读）
  Cargo.toml                 ← 独立 crate（`[workspace]` 空表，与主应用隔离）
  src/
    main.rs                  ← CLI：读 fixtures → 写 tokens.json / demo.html / metrics.json
    text.rs                  ← ICU 分句 + 汉/拉丁脚本切分 + 偏移映射
    ltp.rs                   ← 中文：Electra 分词 + CWS/POS/Dep ONNX
    english.rs               ← 英文：roberta tagger + SuPar parser ONNX
    pipeline.rs              ← 组合两种语言、派生标记
    eisner.rs                ← 依存解码（Eisner / MST），含单元测试
    ortutil.rs / metrics.rs  ← ORT 会话、计时、RSS
  scripts/
    export_ltp.py            ← LTP/small → ONNX（自写，带数值校验）
    export_english_upos.py   ← tagger + SuPar parser → ONNX（带 torch-vs-ONNX 校验，支持 base）
    stage_en_base.py         ← HF snapshot → /workspace/models/en-upos-base（symlink，不复制权重）
    quantize_en_int8.py      ← 单图动态 int8 量化（QUInt8 权重；其它模式仅供复现）
    link_en_variant_assets.py← int8 目录共享 tokenizer/labels/vocab（symlink）
    probe_en_shared_encoder.py ← tagger 与 parser 的 RoBERTa 是否同一份
    check_en_variants.py     ← 变体 vs 同族 fp32 的 tagger/parser 漂移（60 句）
    bench_en_variants.py     ← 4 变体 × spaCy 比较（子进程逐个跑）→ en_compare.md
    en_side_by_side.py       ← base-int8 vs large-int8 小样图
    compare_spacy.py         ← spaCy en_core_web_trf 对照 → compare-en.json
    report_error_cases.py    ← 生成 error-cases.md
    screenshot_demo.py       ← Chrome headless 截图 + 裁掉空白 → demo.png
    gen_eisner_ref.py        ← ltp_core Eisner 的 golden 矩阵（仅比对用）
  fixtures/sentences.json  ← en_compare 共 60 句
  fixtures/en_only.json    ← 只含那 60 句（bench 用，逐句延迟干净）
```

## 1. 复现（导出 + 运行）

```bash
# 0) 模型与缓存
export HF_HOME=/workspace/models/hf-cache

# 1) 中文导出（约 1 分钟；产物在 /workspace/models/ltp-small/onnx/）
/workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/export_ltp.py

# 2) 英文导出（两块，各 ~1.4 GiB；约 3 分钟）
/workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/export_english_upos.py

# 3) Rust CLI（独立 crate；首次编译会下载 ORT 预编译库）
#    模型加载 ~6.5 s、~3.6 GiB RSS，务必加 timeout
timeout 600 cargo run --release --manifest-path spikes/nlp_models/Cargo.toml -- \
  --models-dir /workspace/models \
  --out-dir /workspace/spike-out \
  --input spikes/nlp_models/fixtures/sentences.json \
  --compare-dir /tmp/spike-cmp   # 可选：导出 en_compare 各句，供 compare_spacy.py 用

# 3b) 单元测试（始终加 timeout）
timeout 120 cargo test --release --manifest-path spikes/nlp_models/Cargo.toml

# 4) spaCy 对照 + 出错用例报告
HF_HOME=/workspace/models/hf-cache /workspace/.venv-nlp/bin/python \
  spikes/nlp_models/scripts/compare_spacy.py
/workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/report_error_cases.py

# 4b) 4 变体 bench（子进程逐个跑，写 en_compare_bench.json）+ 并排小样
timeout 1500 /workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/bench_en_variants.py
/workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/en_side_by_side.py

# 5) 截图（Chrome headless 渲染后按内容裁边；宽度 ≤1600）
timeout 180 /workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/screenshot_demo.py \
  --width 1180 --height 5400
```

产物：`/workspace/spike-out/{tokens.json,demo.html,demo.png,metrics.json,compare-en.json,error-cases.md}`，
以及 T-spike-2 的 `en_compare.md`、`en_compare_bench.json`、`check_en_*.json`、
`en_encoder_share_*.json`、`en_side_by_side.{html,png}`、`bench/<variant>/`。
`metrics.json` 含 `en_variant`（名称、目录、tagger/parser/合计字节、分步加载时间）、
`en_latency_ms`（60 句逐句 p50/avg/max）、`peak_rss_mib`（`VmHWM`）。

## 2. 输出格式（`tokens.json`）

每个 token：

```json
{
  "text": "安静", "start": 18, "end": 24,
  "pos": "ADJ",                     // 粗类（着色用）
  "upos_or_ltp_tag": "a",           // 原始标签：LTP 词性 或 UPOS|FEATS
  "head": 11, "deprel": "ATT",      // 句内 1-based head，0 == ROOT
  "is_core_verb": false, "is_connective": false, "is_nonfinite": false,
  "lang": "zh"
}
```

`start/end` 是**原始输入文本的字节偏移**，CLI 会断言
`text[start..end] == token.text`（`offset_errors` 必须为 0，实测 0）。

## 3. 处理流程

1. **分句**：`icu_segmenter` 2.x（`SentenceSegmenter::new(Default::default())`，
   对返回的借用类型调 `segment_str`）。
2. **脚本切分**：每句按字符脚本切成 Han / Latin 段；标点、数字、空白跟随相邻段
   （向前填充，句首的标点回填到后面的段）。
3. **中文段** → LTP ONNX：先跑 CWS 得到分词，再带上 `word_index` 跑第二次得到
   POS + dep 分数 → Rust 用 Eisner 解码 head，按 head 取 rel。
4. **英文段**：同一句里的所有 Latin 段用单个空格拼接，送英文管线；每个词用
   RoBERTa BPE 编码（tagger 走 `tokenizer(sentence)` 风格，parser 复用 esupar
   的 `Field`/`SubwordField` 规则，见 EXPORT.md），token 依次映射回原文偏移。
5. **派生标记**见 §4。

> ⚠️ 中英混排时，**中文段和英文段各自独立解析**（各自一棵依存树），因此
> `head` 只在同一段内可比较；跨语言的「她 quiet…」不是一个连通树。这是刻意的
> 简化，也是阅读标记足够用的粒度。

> ⚠️ **tagger 输入**（2026-11-13 修复）：`english.rs` 用 `tokenizer(" " + sentence)`
> 对**整句**做一次 BPE（esupar 的参考实现），再借 tokenizer 的 `word_ids` 把 BIO
> 标签折到词上；parser 仍按词 `" "+word` 取 piece。`scripts/check_en_variants.py`
> 的复刻实现与此一致——改 tagger 输入时必须同步该脚本，否则变体比较会失真。

## 4. 派生标记规则（T-spike-2 修订）

### `is_core_verb`（句/小句谓语中心）

* **中文**
  1. 根（`head == 0`，LTP `HED`）是动词 `v` → 标为核心；
  2. 根的并列谓语（`COO` 且回溯源到根）→ 也标为核心；
  3. 根是系词（是/为/系/乃）或根不是动词时，取根的 `v`/`a` 子节点作为真正谓语；
  4. 兜底：句中最浅的动词。
  5. **混排句**（同一 sentence 同时含汉字与拉丁字母）：只认词性为动词、或非
     `ATT` 的谓语形容词；系词兜底与"最浅动词"兜底关闭。这样「安静的河边」的
     `安静`（ATT）不会被标成核心谓语。
* **英文**
  1. 纯英文句：UD 把系词挂到谓语名词/形容词上（`She is tired` → `tired:root,
     is:cop`），根本身就是核心；根是系词且其补足语是内容词时，取补足语；
  2. `conj` / `parataxis` 且其 head 是根/核心动词的动词也标为核心。
  3. **混排句**：英文侧只把以 **VERB/AUX** 为中心的谓语标成核心，于是
     `The tired 孩子们…` 里的 `tired`（ADJ root）不再被标为核心谓语；真正的
     动词（`opened`/`closed`/`walk`/`tell`/`睡`）仍标。

### `is_connective`（连词 / 从句标记）

* **英文**（收紧）：
  * 纳入：`CCONJ` / `SCONJ`；依存 `cc` / `cc:preconj` / `mark`；
    关系代词/关系副词：`PronType=Rel`，或 who/whom/whose/which/that 且依存是
    `nsubj/obj/obl/acl/acl:relcl/nmod/advcl/ref/ccomp/mark`；
    when/where/why(+whenever/wherever) 作相对副词（`advmod/advcl/acl/acl:relcl/mark`）。
  * **排除**：介词（`ADP`，如 about/in/on/as —— 即使 parser 偶尔给了 `mark`；
    仅当 tagger 明确标 `ExtPos=SCONJ` 时才认）；**不定式 `to`**（永远是 false）。
* **中文**：词性 `c`，或 `ZH_CONNECTIVES` 表（因为/所以/虽然/但是/如果/那么/
  不仅/而且/尽管/即使/无论/不管/只要/只有/于是/因此/然而…）。介词 `p` 一律
  不标；单独的「以」不标（它只是 `以` 作为介词/构词成分时）。

### 词性着色（粗类）

* 中文 `zh_coarse`：`r`（代词：她/我们/谁）→ **PRON**，不再映射为 ADV；
  PRON 不在 NOUN/VERB/ADJ/ADV 四个着色类里，因此**默认不着色**。
* 英文 `en_coarse`：PRON/DET/ADP 等 → OTHER，不着色。
* `upos_or_ltp_tag` 始终保存原始标签（LTP 词性或 UPOS\|FEATS）。

### `is_nonfinite`（仅英文）

* `VerbForm=Inf|Ger|Part`；`TO` + 不定式动词（tagger 偶尔漏 `VerbForm`）；
  `xcomp` 且非 `Fin`。
* 不标：核心限定谓语；`cop` / `aux` 上的 `be`/`have`（"can be dangerous" 的
  `be` 偶尔带 Part 特征，但它是助动词）。
* 纯形容词用法的分词（`sleeping village`，标签 `ADJ` 无 `VerbForm` 且 deprel
  `amod`）不标；「verbally 使用」的分词（`Exhausted from the trip, …` 的
  `Exhausted` 带 `VerbForm=Part`）标。**已知限制**：`have been working` 这类
  助动词链里的分词会被标（规则只排除 `cop/aux` 直接依附）。

### 演示样式（`demo.html` / `demo.png`）

* 连词：词下方**实线**下划线（`border-bottom:2px solid`）。
* 英文非谓语：词下方**虚线**下划线（`border-bottom:2px dotted`）。
* 核心谓语：词的**下方一个圆点**（英文每个核心词一个 `::after` 圆点；中文每个
  汉字下方一个 `text-emphasis: filled dot` + `text-emphasis-position: under`，
  按字包 `span.corez`）。不再使用会打到字上方的 `text-emphasis`，也不给英文
  每个字母加点。同时有下划线的词，圆点被推到下划线再下方（`bottom:-14px`），
  `line-height:2.35` + `padding-bottom` 防止裁切。
* 图例文字：「下方圆点 / 下方虚线」。

## 5. 平台支持（`ort` 2.0.0-rc.15，已核对 `ort-sys` 的 `build/download/dist.tsv`）

| 目标 | 是否有官方预编译 ONNX Runtime | 建议 |
|---|---|---|
| `x86_64-unknown-linux-gnu` | ✅ (`none` 纯 CPU 分发) | 本 spike 的开发平台 |
| `aarch64-apple-darwin`（Apple Silicon） | ✅ 但只有 `coreml` / `coreml,webgpu` 分发 | 开启 `features=["coreml"]`，或加 `lax-feature-matching` 用 coreml 构建纯 CPU 也可 |
| **`x86_64-apple-darwin`（Intel Mac）** | ❌ **无预编译分发** | 必须自备 `libonnxruntime.dylib`，用 `load-dynamic` 特性在运行时加载并随 app 打包 |
| `x86_64-pc-windows-msvc` | ✅ `directml` / `webgpu` / `cuda…` 分发 | 打包时把 `onnxruntime.dll` 放到 exe 旁；`ort` 的 `copy-dylibs` 特性会自动拷贝 |
| `aarch64-pc-windows-msvc` | ✅ `directml` 分发 | 同上 |

说明：

* `ort` 默认开启 `download-binaries` + `copy-dylibs`：构建时下载官方库，
  并把 dylib/dll 复制到 `target/<profile>/`，**运行时不需要额外配置**。
* 应用发布（quiet-md）建议：不要依赖构建期下载，改为 **随包提供**
  `onnxruntime.{dylib,dll}` + `ort` 的 `load-dynamic` 特性（运行时
  `ort::init_from(path)`），这样用户机器无需网络、也不必匹配构建机的版本。
* Intel Mac 是唯一需要「自己找库」的目标；官方 1.31 release 在
  `microsoft/onnxruntime` GitHub Releases 提供 `onnxruntime-osx-x86_64-*`
  的 dylib，可直接打包。

## 5b. 与 spaCy `en_core_web_trf` 的一致性（`compare-en.json`）

在 fixtures 的 `en_compare` 英文对照句上（当前 60 句），逐词对齐后：

| 指标 | 结果 |
|---|---|
| 粗类词性一致率（名词/动词/形容词/副词） | **98.1%**（309/315） |
| 依存 root 一致率（同一 token 作根） | **78.3%**（47/60） |
| 谓语中心一致率（`is_core_verb` vs spaCy ROOT/acomp） | **96.7%**（58/60） |

不一致主要来自 UD 与 spaCy 的清规不同（系词句 root 取舍、`obl`/`advmod` 介词），
以及 spaCy 对 `don't` 之类的切分。逐句明细见 `compare-en.json` 的 `per_sentence`。

## 6. 许可

* **LTP/small**（HIT-SCIR）：模型权重随 `LTP/small` 发布，个人自用场景可接受；
  商用/再分发前请核对仓库许可（Apache-2.0 相关条款），并遵守其引用要求。
* **KoichiYasuoka/roberta-large-english-upos**：模型卡标注 `cc-by-sa-4.0`，
  基于 `FacebookAI/roberta-large`（MIT）。**CC-BY-SA 有传染性**，若要随
  闭源应用分发该权重，需要研究替代模型或单独授权。
* `esupar.model`：SuPar（MIT）训练的权重，随上述模型卡一同发布；同样受
  `cc-by-sa-4.0` 约束。
* 本 spike 的 Rust 代码：与仓库一致（MIT）。
* **权重一律不进 git**（`.gitignore` 已忽略 `*.onnx` / `*.data` / `*.safetensors`
  / `*.bin`），只提交导出脚本与标签 JSON。

## 7. 已知限制

* 中文与英文各自独立解析（见 §3）。混排时的谓语误标已在 T-spike-2 修正
  （见 §4；`tired` / 「安静」不再被标核心谓语），但混排句的英文侧只认
  VERB/AUX 根，纯英文片段里合法的系词句（形容词 root）在混排句中不会标核心。
* `demo.html` 按 group 原文的字节空隙回插空白，避免英文 token 连写。
* 英文 parser 内嵌第二份 roberta-large，体积大（见 EXPORT.md §3）。
* **依存解码不是投射 Eisner**：`decode_eisner` 目前是「贪心最优 head + 破环
  + 修复单根」的解码器（`decode_mst`）。原因见 EXPORT.md §5：
  `ltp_core.algorithms.eisner` 的 `stripe()` 把 `C[i,r]` 与 `C[j-1-r,j]` 配对，
  并非教科书递推；我们尝试过的投射 DP 版本过不了对拍与穷举最优性测试，因此
  没有发布。实测 fixtures 的英文 3 句与 `esupar` 官方 head 逐词一致，
  中文与 LTP 的 head 在绝大多数词上一致（见 EXPORT.md §5）。
  `cargo test` 覆盖：随机矩阵必为合法树、不会优于穷举最优解。
* CWS 分词在极少数句子会与 LTP Python `pipeline()` 的最终分词略有差异
  （Python 端有多一步 `Hook` 重分段，ONNX 刻意不导出）；POS/依存不受影响。
* LTP 粗类映射（`pipeline.rs::zh_coarse`）：`z`（轻轻地、金黄）着形容词色；
  `r`（代词：她/我们）→ PRON、不着色（T-spike-2 修正，旧版误当 ADV）。
* 英文非谓语：助动词链中的分词（`have been working`、`is replaced`）会被标成
  非谓语；`Having finished dinner` 的 `finished` 被模型标为限定式，因此漏标。
  见 `en_compare.md` §2 的对/错例。
* int8：`scripts/quantize_en_int8.py` 默认 QUInt8 权重；QInt8 会明显劣化
  （见 EXPORT.md §7）。
* LTP 会把「孩子们」切成「孩子」+「们」（`k`，后缀），与 LTP 一致；旧原型
  的问题是把「孩子」整词错标，现已纠正为名词 `n`。
