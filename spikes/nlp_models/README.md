# `spikes/nlp_models` — 离线神经词性 + 依存 spike（T-spike-1）

验证：**完全离线**的中英词性标注 + 依存分析能否用 **Rust + `ort`（ONNX
Runtime）在 CPU 上**跑通，并据此派生阅读标记（名词/动词/形容词/副词着色、
连词下划线、英文非谓语虚线、核心谓语着重号）。

中文用 **LTP/small**（ltp-core），英文用
**KoichiYasuoka/roberta-large-english-upos**（UPOS）+ 它在 `esupar.model` 里
捆绑的 **SuPar biaffine 依存**。spaCy `en_core_web_trf` **只作对照**。

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
    export_english_upos.py   ← tagger + SuPar parser → ONNX（带数值校验）
    compare_spacy.py         ← spaCy en_core_web_trf 对照 → compare-en.json
    report_error_cases.py    ← 生成 error-cases.md
    screenshot_demo.py       ← Chrome headless 截图 + 裁掉空白 → demo.png
    gen_eisner_ref.py        ← ltp_core Eisner 的 golden 矩阵（仅比对用）
  fixtures/sentences.json
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
  --compare-dir /tmp/spike-cmp   # 可选：导出 20 句英文，供 compare_spacy.py 用

# 3b) 单元测试（始终加 timeout）
timeout 120 cargo test --release --manifest-path spikes/nlp_models/Cargo.toml

# 4) spaCy 对照 + 出错用例报告
HF_HOME=/workspace/models/hf-cache /workspace/.venv-nlp/bin/python \
  spikes/nlp_models/scripts/compare_spacy.py
/workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/report_error_cases.py

# 5) 截图（Chrome headless 渲染后按内容裁边）
/workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/screenshot_demo.py
```

产物：`/workspace/spike-out/{tokens.json,demo.html,demo.png,metrics.json,compare-en.json,error-cases.md}`。

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

## 4. 派生标记规则

### `is_core_verb`（句/小句谓语中心）

* **中文**
  1. 根（`head == 0`，LTP `HED`）是动词 `v` → 标为核心；
  2. 根的并列谓语（`COO` 且回溯源到根）→ 也标为核心；
  3. 根是系词（是/为/系/乃）或根不是动词时，取根的 `v`/`a` 子节点作为真正谓语；
  4. 兜底：句中最浅的动词。
* **英文**
  1. UD 把系词挂到谓语名词/形容词上（`She is tired` → `tired:root, is:cop`），
     所以**根本身就是核心**；若根是动词而其 `cop/aux` 链上还有内容词，取补足语；
  2. `conj` / `parataxis` 且其 head 是根/核心动词的动词也标为核心；
  3. 没有根时退化为「所有动词」。

### `is_connective`（连词 / 从句标记）

* **英文**：`CCONJ`、`SCONJ`、依存 `mark`/`cc`/`cc:preconj`；
  关系词：`PronType=Rel`，或 that/which/who/whom/whose 且依存是
  `nsubj/obj/obl/acl/acl:relcl`；`DET+PronType=Rel`。
* **中文**：LTP 词性 `c`，加上关联词表
  （因为/所以/虽然/但是/如果/那么/不仅/而且/尽管/即使/无论/不管/只要/只有/
  于是/因此/然而…，见 `src/pipeline.rs::ZH_CONNECTIVES`）。
  **介词 `p` 不标**（避免把所有介词都当连词）。

### `is_nonfinite`（仅英文）

`VerbForm=Inf|Ger|Part` 的词，或 `TO + VB` 的不定式；**根上的限定谓语永远不标**
（`VerbForm=Fin` 或核心谓语的词会被强制清成 false）。

> 注意：本模型把 `sleeping village` 的 `sleeping` 标成 `ADJ|Degree=Pos`
> （与 UD 标注一致），所以它**不会**被标成 nonfinite。这是模型行为，不是 bug。

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

* 中文与英文各自独立解析（见 §3）。混排时名词性中文片段里的形容词
  （如「安静的河边」的「安静」）或英文系词句的 root 形容词（`tired`）
  可能被标成核心谓语。
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
  `r`（代词：她/我们）着副词色（LTP 标签体系里 `r` 与 `d` 并列），这是
  一个已知的颜色瑕疵，原始标签仍在 `upos_or_ltp_tag` 里。
* LTP 会把「孩子们」切成「孩子」+「们」（`k`，后缀），与 LTP 一致；旧原型
  的问题是把「孩子」整词错标，现已纠正为名词 `n`。
