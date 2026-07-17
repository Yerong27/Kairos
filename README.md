# Kairos AI Job Agent

Kairos 是一个本地运行的求职匹配助手。它通过 Chrome extension 读取当前职位页面，结合你的简历，用 Gemini 提取职位要求并计算匹配度，最后把完整分析写入 Notion 数据库。

## 日常使用：只看这一节就够了

### 1. 启动后端

打开 Terminal，执行：

```bash
cd Kairos
source .venv/bin/activate
uvicorn main:app --reload
```

看到类似下面的内容就表示启动成功：

```text
Uvicorn running on http://127.0.0.1:8000
```

不要关闭这个 Terminal 窗口。Kairos 使用期间，后端必须一直运行。

可以打开 <http://127.0.0.1:8000/health> 检查状态。正常结果应包含：

```json
{"status":"operational","backend":"connected","v3_available":true}
```

### 2. 打开 Chrome extension

如果 extension 已安装，点击 Chrome 工具栏中的 **JD Extractor (Local)** 图标即可。

如果还没安装：

1. 在 Chrome 打开 `chrome://extensions/`。
2. 打开右上角的 **Developer mode（开发者模式）**。
3. 点击 **Load unpacked（加载已解压的扩展程序）**。
4. 选择项目里的 `browser-extension` 文件夹。
5. 建议把 **JD Extractor (Local)** 固定到工具栏。

修改 extension 代码后，需要回到 `chrome://extensions/`，点击该 extension 的刷新按钮。

### 3. 按 extension 的三个步骤操作

#### Step 1 — Connect Notion

1. 点击 **Connect Notion**。
2. 在 Notion 授权页面选择包含目标数据库的页面或数据库。
3. 授权后，在 Kairos 的数据库列表中点击 **Kairos-tracker**。
4. 页面显示 **Notion Connected** 后，extension 会自动保存连接并关闭该页面。

如果看到 **No databases found**，通常不是选错了，而是 Notion 在 OAuth 后还没有完成搜索索引。等待几秒后直接刷新当前页面；不要急着重新配置 integration。

#### Step 2 — Upload Resume

1. 选择 `.pdf` 或 `.txt` 简历。
2. 点击 **Upload**。
3. 看到 **Upload complete** 即可。

简历只需上传一次，内容会保存在本机的 `data/notion_oauth.db` 中。更换简历时重新上传即可。

PDF 必须含有可复制的文字。纯扫描图片 PDF 目前无法 OCR，可能会报 `Resume text too short`。

#### Step 3 — Analyze This Page

1. 在 Chrome 打开一个职位详情页面。
2. 等待职位页面加载完成；Kairos 会自动尝试点击 job description 的 **Show more**。
3. 点击 extension 图标。
4. 点击 **Analyze this page**。
5. extension 会先显示提取到的 JD 字符数、来源和质量，然后开始分析。
6. 打开 Notion 的 **Kairos-tracker**，查看新生成的职位分析页面。

Gemini 分析和 Notion 写入可能需要一些时间。保持 extension 弹窗打开时，它会显示成功或后端返回的具体错误。模型限额、网络错误或要求提取为空时，Kairos 不会再写入 `Error (Degraded)` 页面。

## 首次安装

### 环境要求

- macOS、Linux 或 Windows
- Python 3.10 或更高版本
- Google Chrome
- Gemini API key
- 一个可用的 Notion public integration
- 一个 Notion 数据库

### 创建 Python 环境

首次安装执行：

```bash
git clone https://github.com/Yerong27/Kairos.git
cd Kairos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 配置 `.env`

复制示例配置，然后填写真实凭据：

```bash
cp .env.example .env
```

主要配置如下：

```dotenv
GEMINI_API_KEY=你的_Gemini_API_Key
GEMINI_MODEL=gemini-2.5-flash-lite

NOTION_CLIENT_ID=你的_Notion_Client_ID
NOTION_CLIENT_SECRET=你的_Notion_Client_Secret
NOTION_REDIRECT_URI=http://localhost:8000/notion/callback

KAIROS_DISABLE_CACHE=false
KAIROS_DEBUG_DUMP=false
KAIROS_RUN_ID=run1
```

注意：

- 不要把真实 API key、Notion secret 或 `.env` 发给别人。
- Notion integration 中配置的 redirect URI 必须与 `.env` 完全一致。
- extension 固定连接本机 `127.0.0.1:8000`，因此默认端口必须是 `8000`。
- 本地数据固定保存在项目根目录的 `data/`，该目录已被 Git 忽略。

### Notion integration 设置

这个项目使用 Notion OAuth，而不是让 extension 直接保存一个固定 token。

Notion integration 至少需要：

- 读取内容，用来搜索已授权数据库。
- 插入内容，用来创建分析页面。
- OAuth redirect URI：`http://localhost:8000/notion/callback`。

授权时必须把目标数据库分享给该 connection。当前使用的数据库名是 **Kairos-tracker**。

### Notion 数据库字段

推荐在 **Kairos-tracker** 中创建以下字段，并确保名称和类型完全一致：

| 字段名 | Notion 类型 | 必需程度 |
| --- | --- | --- |
| Job Title | Title | 必需 |
| Company | Text | 推荐 |
| Match Score | Number | 推荐 |
| Should Apply? | Select | 推荐 |
| Seniority Gap | Select | 推荐 |
| Missing Skills | Multi-select | 推荐 |
| Required Skills | Multi-select | 推荐 |
| URL | URL | 推荐 |

如果推荐字段不存在或类型不匹配，程序会尝试进入 Safe Mode，只写入 `Job Title` 和页面正文。因此 `Job Title` 这个 Title 字段尤其重要。

## 它是怎么工作的

```text
当前 LinkedIn 职位网页
    ↓ JSON-LD / Shadow DOM / LinkedIn selectors / fallback
清洗后的职位描述（最多 18,000 字符）
本地 FastAPI 后端
    ↓ Gemini 提取职位、领域、工具和 seniority
确定性评分引擎
    ↓ 与已上传简历进行匹配
Notion writer
    ↓
Kairos-tracker 中的完整 JD ↔ Resume 要求矩阵
```

主要组成：

- `main.py`：后端入口、Notion OAuth、简历上传、分析 API 和本地缓存。
- `browser-extension/`：Chrome extension 界面、网页文字提取和 API 调用。
- `backend/llm/analyze_v3.py`：Gemini 职位信息提取。
- `backend/scoring/scoring_engine_v3.py`：匹配度、seniority gap 和申请建议。
- `backend/notion/writer.py`：把结果格式化并写入 Notion。
- `backend/config/`：领域和工具分类配置。
- `data/notion_oauth.db`：本地授权、所选数据库和简历数据。
- `data/v3_cache/`：相同职位和简历的分析缓存。
- `debug_runs/`：开启 debug 后生成的诊断结果。
- `archive/`：旧版本备份，不是当前运行入口。

Notion 的核心输出不是大模型主观挑选的几个 strengths/gaps，而是对已验证 JD 要求逐项展示：

- `must / should / nice-to-have`
- `matched / partial / missing / unverified`
- JD 原文证据
- 实际支持匹配的简历证据
- JD 明确提到的工具及其匹配状态

`missing` 只表示简历中没有找到证据。不要为了关键词而添加不真实的技能；如果确实具备该能力，应把对应项目、职责或结果更明确地写入简历。

## 常见问题

### `Backend not reachable at 127.0.0.1:8000`

后端没有运行，或启动端口不是 8000。回到 Terminal，按“启动后端”一节重新启动。

如果提示端口已被占用，先打开 <http://127.0.0.1:8000/health>。如果能看到正常 JSON，说明后端已经在运行，不要再启动第二个。

### `No databases found`

Notion 官方说明，OAuth 刚完成时搜索索引可能不是即时更新的。等待几秒，刷新数据库选择页面。当前授权已经确认可以读取到 `Kairos-tracker`。

如果多次刷新仍为空：

1. 确认授权的是正确 Notion workspace。
2. 打开 `Kairos-tracker`，在页面菜单的 Connections 中确认 Kairos integration 有访问权限。
3. 回到 extension，点击 **Change database** 重新授权。

### `Notion OAuth not configured`

`.env` 中缺少 `NOTION_CLIENT_ID` 或 `NOTION_CLIENT_SECRET`，或者后端不是从项目根目录启动。修改后要重启后端。

### `Invalid or expired OAuth state`

你可能刷新了旧的 callback 页面，或后端在授权过程中重启。关闭该页面，再从 extension 点击 **Connect Notion**。

### extension 一直显示 `Checking...` 或连接状态没有更新

1. 确认后端在运行。
2. 关闭并重新打开 extension 弹窗。
3. 在 `chrome://extensions/` 刷新 extension。
4. 必要时重新点击 **Connect Notion**。

### 简历上传失败

- 只支持 `.txt` 和 `.pdf`。
- PDF 必须含有可提取文字。
- 简历提取出的文字必须至少 30 个字符。
- 检查 Terminal 中的具体错误。

### 点击 Analyze 后 Notion 没有新页面

先看运行后端的 Terminal。常见原因包括：

- `GEMINI_API_KEY` 无效、额度不足或网络不可用。
- 职位页面没有完整加载，或者网页禁止 extension 注入。
- 当前页面不是普通网页，例如 `chrome://` 页面或 Chrome Web Store。
- Notion 数据库字段名称或类型不匹配。
- Notion 授权已经失效。

extension 的 `Request sent` 只表示提取请求已经开始，不代表 Gemini 和 Notion 写入已经完成。

如果错误中包含 Gemini `429` 或 `quota exceeded`，表示 API key 当前额度已用完。Kairos 会停止本次分析且不写入 Notion；额度恢复或更换有额度的 key 后重试。

### 修改简历或逻辑后，结果还是旧的

分析会保存在 `data/v3_cache/`。缓存 key 使用清洗后的职位文本、简历、标题和输出语言；LinkedIn 的发布时间或申请人数变化不会轻易导致重复分析。更换简历后会自动产生新结果，不完整或 degraded 的结果不会缓存。

调试时可以在 `.env` 设置：

```dotenv
KAIROS_DISABLE_CACHE=true
```

修改后重启后端。调试结束建议改回 `false`，避免重复调用 Gemini。

## 停止和重新启动

停止后端：在运行 Uvicorn 的 Terminal 中按 `Control + C`。

下次使用只需再次执行：

```bash
cd Kairos
source .venv/bin/activate
uvicorn main:app --reload
```

一般不需要重新安装 extension、重新连接 Notion 或重新上传简历。

## 开发与测试（可选）

安装开发和测试依赖：

```bash
python -m pip install -r requirements-dev.txt
```

运行不调用真实 Notion 写入的单元测试：

```bash
python -m pytest tests
```

## 当前已知限制

- extension 只有在弹窗保持打开时才会显示最终成功或错误状态。
- Notion 首次 OAuth 后仍可能有短暂索引延迟；数据库选择页提供 Refresh 按钮。
- 网页提取依赖页面 DOM；复杂的动态网站可能提取不完整。
- 只支持文字型 PDF，不支持扫描件 OCR。
- 当前 Notion API 默认版本仍是 `2022-06-28`，未来应迁移到新的 `data_source` API。
- OAuth token、简历内容和数据库选择保存在本机 SQLite 文件中，没有额外加密；不要把 `data/notion_oauth.db` 分享或提交到公开仓库。
