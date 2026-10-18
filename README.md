# 澄明 · 财报研究助手（Financial RAG Assistant）

基于 **MinerU + Chroma + 通义千问（DashScope）** 的上市公司财报问答助手。将年报 PDF 解析为文字与表格，建立向量知识库，通过「向量 + BM25」混合检索找到证据，再由大模型严格依据证据作答，并给出引用来源。

> 示例数据：厦门灿坤实业股份有限公司 2019 年年度报告（已附带 MinerU 解析结果，见 `pdf_processed/`）。

## 功能

- **PDF 解析**：使用 MinerU（pipeline 后端）提取正文、标题与表格（HTML），保留页码。
- **结构化分块**：按标题切分正文；表格单独成块，超长表格按行切分并保留表头。
- **混合检索**（`rag_retrieval.py`）：向量召回 + 中文二元词 BM25，经 RRF 融合；命中章节标题时自动补齐同节正文和同表的相邻行，并对重复片段去重。
- **证据约束回答**：提示词要求只依据检索片段作答，信息不足时明确说明“无法回答”。
- **三种使用方式**：命令行演示、只读 JSON CLI（`search` / `ask`）、本地网页界面（FastAPI）。
- **离线回退**：没有向量库时，`search` 可直接在已解析的 JSON 上做本地 BM25 检索，无需 API Key。

## 架构

```
my_pdfs/*.pdf ──MinerU──▶ *_content_list.json ──分块──▶ DashScope Embedding ──▶ Chroma (my_chroma_db)
                                                                                    │
用户问题 ──▶ Embedding + BM25 ──▶ RRF 融合 + 上下文补齐 ◀──────────────────────────────┘
                     │
                     ▼
            Qwen 生成回答（附来源） ──▶ CLI / Web UI
```

## 目录结构

```
├── financial_assistant.py   # 核心：MinerURAGSystem（解析/分块/建库/检索）、RAGQASystem（问答）、CLI
├── rag_retrieval.py         # 混合检索与章节上下文补齐
├── rag_llm.py               # 回答模型调用、限流重试与错误提示
├── web_app.py               # FastAPI 后端（/api/status, /api/initialize, /api/ask）
├── launch_web.py            # 启动本地服务并自动打开浏览器
├── static/                  # 网页前端（HTML/CSS/JS）
├── skills/financial-rag/    # 通用 Agent Skill：通过 CLI 查询本地财报知识库
├── tests/                   # 单元测试（不访问网络）
├── pdf_processed/           # 示例：MinerU 解析结果 *_content_list.json
├── my_pdfs/                 # 放入待解析的 PDF（不入库）
├── processed/               # MinerU 完整输出目录（不入库）
├── requirements.txt         # 网页 / 检索 / 问答依赖
├── requirements-mineru.txt  # 可选：解析新 PDF 所需的 MinerU、PyTorch
├── install.bat / start.bat  # Windows 一键安装 / 启动
└── .env.example             # 环境变量模板
```

## 快速开始

需要 Python 3.11 和一个 [阿里云百炼（DashScope）](https://bailian.console.aliyun.com/) API Key。

```bash
git clone <your-repo-url>
cd financial-rag-assistant
python -m venv .venv-web
# Windows: .venv-web\Scripts\activate    macOS/Linux: source .venv-web/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 然后填写 DASHSCOPE_API_KEY
```

Windows 用户也可以直接双击 `install.bat` 安装、`start.bat` 启动。

### 1. 网页界面

```bash
python launch_web.py
```

浏览器打开 `http://127.0.0.1:8765`，点击“初始化知识库”。首次初始化会用 `pdf_processed/` 或 `processed/` 下的解析结果建立向量库（调用 Embedding API），之后即可提问。服务只监听本机。

### 2. 只读 CLI（JSON 输出）

```bash
# 只检索证据
python financial_assistant.py search --question "2019年一季度归属于上市公司股东的净利润是多少？" --top-k 5
# 检索 + 生成回答
python financial_assistant.py ask --question "2019年度营业收入是多少？"
```

`retrieval_mode` 为 `chroma_hybrid`（已有向量库）或 `parsed_files_bm25`（离线回退）。CLI 不会解析 PDF，也不会修改向量库。
失败时输出 `{"ok": false, "error": "中文原因"}`，退出码为 1。

### 3. 演示流程（完整管线）

```bash
python financial_assistant.py
```

依次执行：检查 `my_pdfs/` → 若无解析结果则用 MinerU 解析 → 分块 → **清空并重建** `my_chroma_db` → 回答一个示例问题。解析新 PDF 需先安装 `pip install -r requirements-mineru.txt`。

### 4. 作为 AI 代理的 Skill 使用

仓库自带一个通用 Skill：`skills/financial-rag/`（`SKILL.md` + 转接脚本）。它不依赖任何特定代理，实际调用的就是上面的 `search` / `ask` 命令。

1. 按“快速开始”装好依赖，并在 `.env` 中填写 `DASHSCOPE_API_KEY`（只用 `search` 的离线检索时可不填）；
2. 把 `skills/financial-rag/` 复制（或链接）到你所用代理读取 Skill 的目录，例如 Claude Code 为项目下的 `.claude/skills/`，其他支持 `SKILL.md` 的代理请按其文档放置；
3. 直接提问即可，例如“用 financial-rag 查一下公司 2019 年面临哪些风险”。

转接脚本按以下顺序确定项目根目录：`--project-dir` 参数 → 环境变量 `FINANCIAL_RAG_PROJECT` → 从脚本位置逐级向上查找 `financial_assistant.py`。所以只要 Skill 放在项目目录内（无论哪个子目录），都能自动找到项目；放在项目之外（例如用户级 skills 目录）时，需要设置 `FINANCIAL_RAG_PROJECT=<本项目根目录>` 或运行时加 `--project-dir`。

不支持 Skill 的代理或脚本，直接调用命令行即可，两种写法结果完全相同：

```bash
python financial_assistant.py search --question "公司面临哪些风险？"
python skills/financial-rag/scripts/financial_rag.py ask --question "公司面临哪些风险？"
```

| 命令 | 作用 | 需要 API Key | 产生费用 |
| --- | --- | --- | --- |
| `search` | 返回检索到的财报原文证据 | 有向量库时需要；离线检索不需要 | 有向量库时少量 embedding 费用 |
| `ask` | 检索后由大模型生成回答 | 需要 | 是 |

### 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `模型服务的 HTTPS 连接失败` | 网页端和命令行默认让 DashScope 请求绕过系统代理直连，开着 Clash 等代理也能用，一般不会出现。若出现，请检查网络；如果你的网络必须经代理访问外网，在 `.env` 中设置 `DASHSCOPE_USE_SYSTEM_PROXY=1`，并确认代理能访问 `dashscope.aliyuncs.com`。 |
| `模型调用受限（429）` | 模型限流或额度不足。稍后重试，或在 `.env` 中换用其他模型（`LLM_MODEL`）。 |
| `API 密钥认证失败` | 检查 `DASHSCOPE_API_KEY` 是否正确。 |
| `缺少运行依赖 xxx` | 未安装依赖，运行 `pip install -r requirements.txt`。 |
| `向量索引文件无法读取` | Windows 下 Chroma 不能读取含中文的路径。把项目放在纯英文路径下，或改用网页端（会自动迁移索引）。 |

## 配置项（`.env`）

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | DashScope API Key | 必填 |
| `EMBEDDING_MODEL` | 向量模型 | `text-embedding-v2` |
| `LLM_MODEL` | 回答模型，如 `qwen-flash`、`qwen-plus` | `qwen-flash` |
| `REPORT_TITLE` | 提示词中的财报名称 | 灿坤 2019 年报 |
| `CHENGMING_DATA_DIR` | Windows 下项目路径含中文时，向量库的纯英文存放目录 | `%LOCALAPPDATA%/ChengmingFinancialAssistant` |
| `DASHSCOPE_USE_SYSTEM_PROXY` | 默认 DashScope 请求绕过系统代理直连（只影响本程序进程，不改系统设置）；设为 `1` 时改走系统代理 | 直连 |

## 更换财报

1. 把新 PDF 放入 `my_pdfs/`，删除 `processed/`、`pdf_processed/` 下旧的解析结果；
2. 安装 `requirements-mineru.txt`，运行 `python financial_assistant.py` 解析并重建向量库；
3. 在 `.env` 中修改 `REPORT_TITLE`。

## 测试

```bash
python -m unittest discover -s tests
```

测试使用内存假数据与 mock，不访问网络，也不会写入向量库。

## 注意事项

- Windows 下 Chroma 的 HNSW 索引无法读取含中文的路径；网页端会自动把向量库迁移到英文目录（原库保留）。
- 回答来自检索与模型生成，可能存在偏差，请以原始财报为准；“检索距离”不是可信度。
- 本项目仅用于学习与研究，不构成任何投资建议。
