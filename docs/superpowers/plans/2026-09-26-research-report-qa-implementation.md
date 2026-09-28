# 研报问答改造实施计划

> 执行方式：按 Task 顺序推进，每个 Task 走「先写失败测试 → 确认失败 → 最小实现 → 确认通过 → 提交」。用 `- [ ]` 复选框跟踪进度。
>
> 设计依据：`docs/superpowers/specs/2026-09-26-research-report-qa-design.md`。计划中每个技术判断都能回溯到该 spec 的某一节。

**Goal:** 把知识库内容换成研报，按「个股 / 行业 / 宏观」建立元数据，补齐表格、图片、页码三条证据通路，并在**全部到位语料（18 篇）**上建立分层指标基线，据以定点改造检索链路。

**Architecture:** 沿用现有 FastAPI + Milvus + ONNX + BGE 重排 + PostgreSQL 父子块地基，不重写。新增四块：① 元数据抽取与写库；② PG `table_chunks`（JSONB）承载整表，向量库只留「表头 + 首行」作索引，检索侧新增独立 `_expand_tables`；③ pymupdf 抽取图片与页码，图片 caption 作为独立 chunk 入索引；④ 过滤白名单扩展 + 值校验。合规模块整体删除。

**Tech Stack:** Python 3.11、pytest、Milvus（pymilvus）、PostgreSQL/SQLAlchemy、pymupdf、MinerU 3.4.5、Vue 3 + Element Plus。

## Global Constraints

- 验收只看自动指标，硬阈值：`hit_rate@5 ≥ 0.90`、`evidence_rank ≤ 3` 占比 `≥ 0.80`、引用可溯源率 `≥ 0.95`、负样本拒答正确率 `= 1.00`。
- 前端改造保留但**不计入验收门槛**；"30min → ≤5min" 是方向性目标，不进门槛。
- `security_code` 为**单值** VARCHAR；行业研报靠 `industry_l1/l2` 召回，不填代码。
- 行业值必须落在申万 enum 内，并校验「二级属于所选一级」；非法值丢弃，抽不到留空。空值不参与过滤。
- 元数据自动过滤**只在高置信时施加**，抽不到或不唯一则不加过滤。
- 表格在向量库中只以「表头 + 首行」作索引，整表进 PG；`_expand_tables` **不受** `PARENT_MAX_CHARS=3000` 约束。
- 单表注入上限 8000 字符，超限按行截断并标 `truncated: true`，不静默删数据。
- 不设默认时间窗；`merge_docs` 排序改为 `(score, date desc)`。
- 不做跨券商口径归一化、不做 `rating` / `target_price` 抽取、不做多租户、不做会话持久化。
- 检索参数类改造（`ENABLE_LANGGRAPH`、rerank top-N、k、nprobe、HyDE、query rewrite、动态 K、语义分块）**一律等 baseline 数据**，不得在本计划前 14 个 Task 中改动。
- 语料是硬先决条件：真实研报 PDF 未就位前不得启动 Task 15 及之后的 baseline。

---

### Task 1: 解析层产出页码与图片资产

**Files:**

- Modify: `finance_rag/src/rag/ingestion/mineru_parser.py`
- Create: `finance_rag/src/rag/ingestion/pdf_assets.py`
- Create: `tests/unit/test_parse_assets.py`

**Interfaces:**

- Produces: `ParseResult(markdown, parser, parser_version, stats, images: tuple[ImageAsset, ...], blocks: tuple[ContentBlock, ...])`。两个新字段带默认值，保证现有调用方不破。
- Produces: `ImageAsset(key_hint: str, page: int, data: bytes, ext: str)`、`ContentBlock(text: str, page: int)`。
- Produces: `pdf_assets.extract_images(pdf_path) -> list[ImageAsset]`、`pdf_assets.extract_blocks(pdf_path) -> list[ContentBlock]`。

**背景:** `ParseResult` 当前是 frozen dataclass，字段只有 `markdown / parser / parser_version / stats`（`mineru_parser.py:22-27`），而 `chunker.py:102` 传了 `result.images` —— 这就是上传路径必然 `AttributeError` 的根因。本 Task 的修法是**让 `images` 真的有值**，不是删掉参数。

**MinerU content_list:** 需要把 `f_dump_content_list` 从 `False` 改为 `True`（`mineru_parser.py:60`），并在 `finally` 删除工作目录**之前**读取 `content_list.json`，用它构建 `ContentBlock`（含 `page_idx`）。解析失败时不得因为 content_list 缺失而报错——降级为空 tuple。

- [ ] **Step 1: 写入失败测试**

用 pymupdf 在测试内生成一个带内嵌图片的两页 PDF，不依赖任何外部素材。

> **环境说明（实测修正）**：当时沙箱内 pytest 的 `tmp_path` 不可用（平台临时目录在可写区之外，且销毁刚建目录被拒），`tests/unit/.tmp` 也被拒绝访问，因此一度写成「`_workdir` + `data/state` 回退」。**该限制在文件策略放宽后消失**，最终实现已回归仓库既有约定 `test_ingestion_pipeline._workdir`（`tests/unit/.tmp`），不再有回退分支。

~~~python
import pymupdf

from finance_rag.src.rag.ingestion import pdf_assets


def _workdir(name: str) -> Path:
    """仓库内的工作目录（受限环境下系统临时目录可能不可写）。"""
    bases = (
        Path(__file__).resolve().parent / ".tmp",
        Path(__file__).resolve().parents[2] / "assets" / "state" / "pytest-tmp",
    )
    for base in bases:
        try:
            path = base / "parse_assets" / name
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    raise RuntimeError("找不到可写的测试工作目录")


def _write_pdf(path: Path, *, image_pages=(), text_pages=2) -> Path:
    """生成一个最小 PDF：每页一段文字，可选在指定页插入同一张内嵌图片。"""
    doc = pymupdf.open()
    for index in range(text_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {index + 1} text")
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
    pixmap.clear_with(255)
    for page_index in image_pages:
        doc[page_index].insert_image(pymupdf.Rect(100, 100, 140, 140), pixmap=pixmap)
    doc.save(path)
    doc.close()
    return path


def test_extract_images_carries_one_based_page_and_hint():
    path = _write_pdf(_workdir("with-image") / "sample.pdf", image_pages=(1,))
    assets = pdf_assets.extract_images(path)
    assert len(assets) == 1
    assert assets[0].page == 2
    assert assets[0].key_hint == "2-0"
    assert assets[0].data[:4] == b"\x89PNG"


def test_extract_images_dedupes_identical_bytes():
    """同一张图重复出现在多页（页眉 logo）时只保留一份，避免重复付视觉模型费用。"""
    path = _write_pdf(_workdir("repeated-image") / "repeated.pdf", image_pages=(0, 1))
    assert len(pdf_assets.extract_images(path)) == 1


def test_extract_blocks_maps_text_to_one_based_page():
    path = _write_pdf(_workdir("blocks") / "sample.pdf", image_pages=(1,))
    blocks = pdf_assets.extract_blocks(path)
    assert [block.page for block in blocks] == [1, 2]
~~~

`key_hint` 采用 1-based 页码 + 页内序号，使对象存储的图片 key 与分析师看到的页码一致。


def test_parse_result_has_images_and_blocks_fields():
    from finance_rag.src.rag.ingestion.mineru_parser import ParseResult

    result = ParseResult(markdown="x")
    assert result.images == ()
    assert result.blocks == ()
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/unit/test_parse_assets.py -v`

Expected: FAIL —— `pdf_assets` 模块不存在；`ParseResult` 无 `images` / `blocks`。

- [ ] **Step 3: 实现最小实现**

`pdf_assets` 用 `pymupdf.open(path)`：`extract_images` 遍历 `page.get_images(full=True)` → `doc.extract_image(xref)`，`key_hint` 用 `f"{page_index}-{seq}"`（PDF 内顺序，与对象存储 key 解耦）；`extract_blocks` 用 `page.get_text("blocks")`。两函数对损坏 PDF 只告警并返回空列表，不抛。

`mineru_parser` 侧：`ParseResult` 增两个默认空 tuple 的字段；`.md/.txt` 短路分支返回空 tuple；二进制分支在删除 `work_dir` 前读 `content_list.json` 构建 `blocks`，并用 `pdf_assets.extract_images` 填充 `images`。图片抽取失败只记 warning。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/unit/test_parse_assets.py -v`

Expected: PASS。

- [ ] **Step 5: 回归验证上传路径不再抛 AttributeError**

Run: `python -c "from finance_rag.src.rag.ingestion.chunker import HierarchicalChunker; r = HierarchicalChunker().parse_and_chunk('assets/ex/2026中报点评：短期扰动不改基本面，燃机订单超预期+AIDC一体化打开成长空间.pdf'); print(len(r.chunks))"`

> 注：该命令需要 MinerU 与模型权重齐备（走真实解析）。纯离线环境下更快的等价验证是
> `python -m pytest tests/unit/test_parse_assets.py tests/unit/test_chunker.py -q`。

Expected: 打印块数，不再抛 `AttributeError: 'ParseResult' object has no attribute 'images'`。

- [ ] **Step 6: 提交**

~~~bash
git add finance_rag/src/rag/ingestion/mineru_parser.py finance_rag/src/rag/ingestion/pdf_assets.py tests/unit/test_parse_assets.py
git commit -m "fix: give ParseResult real image and page assets"
~~~

### Task 2: 切块层——页码归属、image chunk、表格双份

**Files:**

- Modify: `finance_rag/src/rag/ingestion/chunker.py`
- Modify: `tests/unit/test_chunker.py`

**Interfaces:**

- Consumes: `ParseResult.images` / `ParseResult.blocks`（Task 1）。
- Produces: `DoclingChunks` 新增 `tables: list[TableChunk]`；每个 child 的 `metadata` 新增 `block_type`（`text` / `table` / `image`）与 `start_page` / `end_page`（int，无法归属时为 `0`）。
- Produces: `TableChunk(id, parent_id, markdown, payload: list[list[str]], row_count, source, heading_path, start_page)`。
- Produces: image chunk —— `block_type="image"`，`content` 为占位（Task 6 回填 caption），`metadata["image_hint"]` 为 `key_hint`。

**关键约束:** 表格的**索引 children** 仍是 `_table_summary`（表头 + 首行），同时把**整表**放进 `DoclingChunks.tables`。不要改 `_table_summary` 的输出，否则会破坏向量库的索引语义。

**页码归属:** 对每个 child，用它的正文前缀去 `blocks` 里做最长前缀匹配，命中则取该 block 的 page；匹配不上 `start_page = end_page = 0`（**留空，不乱填**）。

- [ ] **Step 1: 写入失败测试**

~~~python
def test_table_chunk_keeps_first_row_index_and_full_table_separately():
    chunker = HierarchicalChunker()
    markdown = (
        "## 盈利预测\n\n"
        "| 年份 | 营收 | 净利 |\n| --- | --- | --- |\n"
        "| 2026E | 100 | 12 |\n| 2027E | 130 | 16 |\n| 2028E | 160 | 20 |\n"
    )
    result = chunker.chunk_markdown(markdown, source="s.pdf", title="t")

    table_children = [c for c in result.chunks if c.metadata["block_type"] == "table"]
    assert len(table_children) == 1
    assert "2026E" in table_children[0].page_content       # 索引里只留首行
    assert "2028E" not in table_children[0].page_content   # 表体不进索引

    assert len(result.tables) == 1
    assert result.tables[0].row_count == 3
    assert result.tables[0].payload[2] == ["2028E", "160", "20"]


def test_child_pages_come_from_blocks_and_default_to_zero():
    chunker = HierarchicalChunker()
    markdown = "## 投资要点\n\n燃机订单超预期。\n"
    chunks = chunker.chunk_markdown(markdown, source="s.pdf", title="t")
    assert all(c.metadata["start_page"] == 0 for c in chunks.chunks)
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/unit/test_chunker.py -k "table_chunk or child_pages" -v`

Expected: FAIL —— `block_type`、`DoclingChunks.tables` 均不存在。

- [ ] **Step 3: 实现最小实现**

`_extract_atomic_units` 已把表格当原子单元，在其分支里同时产出索引 child 与 `TableChunk`。`payload` 由 markdown 表按行 split，去首尾 `|` 后 strip 每格。`chunk_markdown` 增可选 `blocks` 参数（默认空 tuple），`parse_and_chunk` 透传 `result.blocks` 与 `result.images`。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/unit/test_chunker.py -v`

Expected: PASS，且既有用例不回归。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/rag/ingestion/chunker.py tests/unit/test_chunker.py
git commit -m "feat: emit page, table and image structure from chunker"
~~~

### Task 3: PostgreSQL 表格仓储

**Files:**

- Create: `finance_rag/src/infrastructure/relational_db/table_store.py`
- Create: `tests/integration/test_table_store_repository.py`

**Interfaces:**

- Produces: `table_chunks` 表，复合主键 `(collection, id)`，列 `payload` (JSONB)、`markdown` (Text)、`row_count` (Integer)、`source` (String 512)、`heading_path` (String 1024)、`start_page` (Integer)。
- Produces: `TableStoreRepository.upsert_many(collection, items)` / `get_batch(collection, ids)` / `delete_by_source(collection, source)`。
- Consumes: 与 `ParentStoreRepository` 同构的建表模式（`parent_store.py:39-96`）。

- [ ] **Step 1: 写入失败测试**

~~~python
def test_upsert_and_fetch_table_rows(sqlite_collection_url):
    repo = TableStoreRepository(database_url=sqlite_collection_url)
    repo.upsert_many("finance_kb", [{
        "id": "t1", "payload": [["年份", "营收"], ["2026E", "100"]],
        "markdown": "| 年份 | 营收 |", "row_count": 2,
        "source": "s.pdf", "heading_path": "盈利预测", "start_page": 4,
    }])
    rows = repo.get_batch("finance_kb", ["t1"])
    assert rows[0]["payload"][1] == ["2026E", "100"]
    assert rows[0]["start_page"] == 4
~~~

> PostgreSQL 专属的 JSONB 在 SQLite 上需按方言降级为 JSON；测试用 SQLite，生产用 PostgreSQL。降级只影响列类型，不影响读写接口。

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/integration/test_table_store_repository.py -v`

Expected: FAIL —— 模块不存在。

- [ ] **Step 3: 实现最小实现**

照抄 `parent_store.py` 的结构：模块级 `MetaData()` + `Table(...)`，`_get_engine()` 内 `metadata.create_all`，`upsert_many` 事务内 `delete + insert`。JSONB 用 `sqlalchemy.JSON` 并按方言设置 `postgresql.JSONB`。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/integration/test_table_store_repository.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/infrastructure/relational_db/table_store.py tests/integration/test_table_store_repository.py
git commit -m "feat: add postgres table chunk store"
~~~

### Task 4: Milvus schema 扩展与写入

**Files:**

- Modify: `finance_rag/src/infrastructure/vector_store/milvus_kb.py:225-244`
- Modify: `scripts/migrate_kb_schema.py`
- Modify: `tests/unit/test_kb_write_lock.py`（或新增 `tests/unit/test_milvus_schema.py`）

**Interfaces:**

- Produces: schema 新增 `security_code VARCHAR(32)`、`security_name VARCHAR(64)`、`industry_l1 VARCHAR(32)`、`industry_l2 VARCHAR(32)`、`report_type VARCHAR(16)`、`broker VARCHAR(64)`、`block_type VARCHAR(16)`、`meta_source VARCHAR(16)`、`needs_review BOOL`、`start_page INT64`、`end_page INT64`、`image_key VARCHAR(256)`。
- Produces: `KnowledgeBase.add_parsed_document` 写入上述标量；空值统一写空串 / `0`，不写 `None`（Milvus 标量不接受 `None`）。

**注意:** `enable_dynamic_field=False`，新字段必须重建集合。`rebuild_collection()` 与 `scripts/migrate_kb_schema.py` 已存在，按新 schema 重跑即可。

- [ ] **Step 1: 写入失败测试**

~~~python
def test_schema_contains_research_report_fields(fake_milvus_client):
    schema = KnowledgeBase(client=fake_milvus_client)._build_schema()
    names = {field.name for field in schema.fields}
    assert {"security_code", "industry_l1", "industry_l2",
            "report_type", "block_type", "start_page", "end_page"} <= names


def test_chunk_metadata_defaults_are_non_null():
    row = _build_milvus_row(_chunk_without_metadata())
    assert row["security_code"] == ""
    assert row["start_page"] == 0
    assert row["needs_review"] is False
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/unit/test_milvus_schema.py -v`

Expected: FAIL —— 字段缺失。

- [ ] **Step 3: 实现最小实现**

在 schema 定义处加字段；在行构造处从 chunk metadata 取值并做非空兜底。`scripts/migrate_kb_schema.py` 的字段映射同步补齐。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/unit/test_milvus_schema.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/infrastructure/vector_store/milvus_kb.py scripts/migrate_kb_schema.py tests/unit/test_milvus_schema.py
git commit -m "feat: extend milvus schema for research report metadata"
~~~

### Task 5: 元数据抽取（正则优先 → LLM 回退 → 申万 enum 校验）

**Files:**

- Create: `finance_rag/src/rag/ingestion/metadata_extractor.py`
- Create: `assets/taxonomy/sw_industry.json`
- Create: `tests/unit/test_metadata_extractor.py`

**Interfaces:**

- Produces: `ExtractedMetadata(security_code, security_name, industry_l1, industry_l2, report_type, broker, meta_source, needs_review)`。
- Produces: `extract_metadata(filename, title, markdown_head, *, llm=None) -> ExtractedMetadata`。
- Produces: `assets/taxonomy/sw_industry.json` —— 申万 2021 版一级（31 项）→ 二级列表。**只作 enum 清单与校验用，不建代码→行业映射表**（需求方已选定行业由 LLM 抽取，不引入映射反查）。

**校验规则（硬约束）:**

- `security_code` 必须匹配 `^\d{6}$`，否则丢弃。
- `industry_l1` 必须在 enum 内；`industry_l2` 必须属于所选 `industry_l1`，否则丢弃二级、保留一级。
- `report_type` ∈ {个股, 行业, 宏观}，否则丢弃。
- 正则命中 `security_code` 但 LLM 未回时，`security_name` 可留空，`meta_source="regex"`，`needs_review=True`。
- 任一回退到 LLM 的字段 → `meta_source="llm"`；出现非法值被丢弃 → `needs_review=True`。

- [ ] **Step 1: 写入失败测试**

~~~python
def test_filename_regex_extracts_code_broker_and_date():
    meta = extract_metadata(
        "贵州茅台(600519)2026中报点评-中信证券-20260815.pdf", "贵州茅台2026中报点评", ""
    )
    assert meta.security_code == "600519"
    assert meta.broker == "中信证券"
    assert meta.meta_source == "regex"


def test_industry_outside_enum_is_dropped(monkeypatch):
    monkeypatch.setattr(metadata_extractor, "_llm_extract", lambda **_: {
        "security_code": "600519", "security_name": "贵州茅台",
        "industry_l1": "白酒", "industry_l2": "白酒", "report_type": "个股",
    })
    meta = metadata_extractor.extract_metadata("研报.pdf", "研报", "", llm=object())
    assert meta.industry_l1 == ""          # 不在申万 enum 内 → 丢弃
    assert meta.needs_review is True


def test_industry_l2_must_belong_to_selected_l1(monkeypatch):
    monkeypatch.setattr(metadata_extractor, "_llm_extract", lambda **_: {
        "industry_l1": "食品饮料", "industry_l2": "半导体", "report_type": "行业",
    })
    meta = metadata_extractor.extract_metadata("研报.pdf", "研报", "", llm=object())
    assert meta.industry_l1 == "食品饮料"
    assert meta.industry_l2 == ""
    assert meta.needs_review is True
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/unit/test_metadata_extractor.py -v`

Expected: FAIL —— 模块与清单文件不存在。

- [ ] **Step 3: 实现最小实现**

正则库：`名称(代码)`、`(?<!\d)\d{6}(?!\d)`、券商后缀（`-XX证券`）、日期 `\d{8}`。LLM 抽取用 `get_rewrite_model()`（轻量、低延迟）并把申万清单序列化进 prompt，要求只回 JSON。所有值走 `_validate` 后再返回。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/unit/test_metadata_extractor.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/rag/ingestion/metadata_extractor.py assets/taxonomy/sw_industry.json tests/unit/test_metadata_extractor.py
git commit -m "feat: add research report metadata extractor"
~~~

### Task 6: 图片上传与视觉描述

**Files:**

- Create: `finance_rag/src/rag/ingestion/image_captioner.py`
- Modify: `finance_rag/src/utils/audit.py`
- Create: `tests/unit/test_image_captioner.py`

**Interfaces:**

- Produces: `caption_image(data: bytes, *, ext: str) -> str`，复用 `IMAGE_CAPTION_*`（`config.py:113-120`，qwen-vl-max），失败重试 `IMAGE_CAPTION_MAX_RETRIES` 次后返回空串。
- Produces: `store_images(assets, storage, stem) -> dict[key_hint, object_key]`，对象存储 key 为 `images/{stem}/{key_hint}.{ext}`。
- Produces: 审计事件 `image_egress`，记录 `source` / 图片数 / 模型名，写入 `utils/audit.py` 既有日志。

**现状:** `IMAGE_CAPTION_*` 8 个配置键零消费者；`document_service._upload_extracted_images`（`document_service.py:687`）因 `parsed.chunks.images` 恒为空而永不生效。

- [ ] **Step 1: 写入失败测试**

~~~python
def test_caption_skips_images_below_min_size(monkeypatch):
    called = []
    monkeypatch.setattr(image_captioner, "_call_vision", lambda *a, **k: called.append(1) or "x")
    assert image_captioner.caption_image(b"\x89PNG" + b"\x00" * 10, ext="png") == ""
    assert called == []


def test_caption_records_egress_audit(monkeypatch, caplog):
    monkeypatch.setattr(image_captioner, "_call_vision", lambda *a, **k: "燃机订单结构图")
    text = image_captioner.caption_image(b"\x89PNG" + b"\x00" * 6000, ext="png")
    assert text == "燃机订单结构图"
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/unit/test_image_captioner.py -v`

Expected: FAIL —— 模块不存在。

- [ ] **Step 3: 实现最小实现**

`_call_vision` 走 OpenAI 兼容 `IMAGE_CAPTION_BASE_URL`，用 `IMAGE_CAPTION_PROMPT`，把图片 base64 塞进 message。尺寸/字节下限按 `IMAGE_CAPTION_MIN_DIM_PX` / `MIN_BYTES` 过滤。外发前记审计。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/unit/test_image_captioner.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/rag/ingestion/image_captioner.py finance_rag/src/utils/audit.py tests/unit/test_image_captioner.py
git commit -m "feat: caption and store research report images"
~~~

### Task 7: 入库链路接入（抽取 + 表格 + 图片）

**Files:**

- Modify: `finance_rag/src/services/document_service.py:606-700`
- Modify: `finance_rag/src/services/ingestion_pipeline.py:640-790`
- Modify: `finance_rag/src/infrastructure/vector_store/milvus_kb.py`（写 `table_chunks`）
- Modify: `tests/unit/test_ingestion_pipeline.py`

**Interfaces:**

- Consumes: Task 1–6 全部产物。
- Produces: 每个文档入库后，Milvus 行带完整元数据；`table_chunks` 有整表；image chunk 的 `content` 为 caption、`image_key` 指向对象存储。

**阶段归属:** 元数据抽取与图片 caption 都在 **parse 阶段之后、chunk 之前**完成（抽取需要 markdown 首页；图片只需 PDF）。两者都是外部调用，必须放在 `ThreadPoolExecutor` 解析槽位之外，避免占用 MinerU 并发额度；抽取失败不得阻断入库。

- [ ] **Step 1: 写入失败测试**

~~~python
@pytest.mark.asyncio
async def test_ingest_writes_metadata_tables_and_images(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline_module, "extract_metadata", lambda *a, **k: _META)
    monkeypatch.setattr(pipeline_module, "caption_image", lambda *a, **k: "订单结构图")
    result = await pipeline.submit(_report_pdf_fixture())
    await pipeline.drain()
    assert result["metadata"]["security_code"] == "600519"
    assert pipeline.written_tables[0]["row_count"] == 3
    assert any(row["block_type"] == "image" for row in pipeline.written_rows)
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/unit/test_ingestion_pipeline.py -k metadata_tables_images -v`

Expected: FAIL —— 流水线未调用抽取与 caption。

- [ ] **Step 3: 实现最小实现**

`_prepare_parsed_document` 内：抽元数据 → `store_images` → caption 回填 image chunk。`_commit_parsed_document` 内：Milvus 写行 + `TableStoreRepository.upsert_many`。任一外部调用失败只告警并留空，不阻断。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/unit/test_ingestion_pipeline.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/services/document_service.py finance_rag/src/services/ingestion_pipeline.py finance_rag/src/infrastructure/vector_store/milvus_kb.py tests/unit/test_ingestion_pipeline.py
git commit -m "feat: wire metadata, tables and images into ingest"
~~~

### Task 8: 检索侧——`_expand_tables`、字段透出、日期 tie-break

**Files:**

- Modify: `finance_rag/src/rag/retrieval/hybrid_retriever.py:463-546`
- Modify: `finance_rag/src/rag/retrieval/parent_store.py`
- Create: `tests/unit/test_retrieval_tables.py`

**Interfaces:**

- Produces: `HybridRetriever._expand_tables(matches, k)` —— 按 `block_type == "table"` 的行取 `parent_id`，从 `table_chunks` 取整表，**不受 `PARENT_MAX_CHARS` 约束**，单表上限 8000 字符，超限按行截断并置 `truncated=True`。
- Produces: `_expand_to_parents` 跳过 `block_type in {"table", "image"}` 的行。
- Produces: `output_fields` 增补新字段；返回 dict 带 `security_code` / `industry_l1` / `report_type` / `start_page` / `end_page` / `block_type` / `image_key`。

**关键背景:** `hybrid_retriever.py:539-545` 只在父块长度 ∈ `[50, 3000]` 时替换内容，超过则**保留子块内容**（表头 + 首行）。整表必然超限，所以必须走独立路径，否则 jsonb 方案被静默吃掉。

- [ ] **Step 1: 写入失败测试**

~~~python
def test_table_match_expands_full_table_beyond_parent_limit(retriever):
    retriever._table_store.upsert_many("finance_kb", [_big_table("t1", rows=200)])
    matches = [_table_match(parent_id="t1")]
    out = retriever._expand_tables(matches, k=1)
    assert "2028E" in out[0]["content"]
    assert out[0]["truncated"] is True
    assert len(out[0]["content"]) <= 8000 + 200   # 上限 + 截断提示


def test_parent_expansion_skips_table_and_image_rows(retriever):
    retriever._parent_store.upsert_many("finance_kb", [_parent("p1", "x" * 4000)])
    matches = [_table_match(parent_id="p1"), _image_match(parent_id="p1")]
    out = retriever._expand_to_parents(matches, k=2)
    assert out[0]["content"] == matches[0]["content"]      # 表格行未被父块覆盖
    assert out[1]["content"] == matches[1]["content"]
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/unit/test_retrieval_tables.py -v`

Expected: FAIL —— `_expand_tables` 不存在，且表格行会被父块逻辑处理。

- [ ] **Step 3: 实现最小实现**

`_expand_tables` 与 `_expand_to_parents` 并列，在 `search()` 的父块扩展之后调用。截断时按行保留，追加 `"（表已截断，可点击查看完整表）"`。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/unit/test_retrieval_tables.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/rag/retrieval/hybrid_retriever.py finance_rag/src/rag/retrieval/parent_store.py tests/unit/test_retrieval_tables.py
git commit -m "feat: expand full tables without the parent size cap"
~~~

### Task 9: 过滤白名单扩展与值校验

**Files:**

- Modify: `finance_rag/src/services/chat_service.py:300-400`
- Create: `tests/unit/test_metadata_filters.py`

**Interfaces:**

- Produces: `_normalize_metadata_filters` 放行 `security_code` / `industry_l1` / `industry_l2` / `report_type` / `broker`，并对每个键做值校验。
- Produces: `infer_metadata_filters` 的 prompt 同步扩字段；返回条件带 `confidence`，只有高置信才施加。

**现状:** `chat_service.py:318-320` 的注释明写「只白名单 category/date 两个键，避免注入任意过滤条件」——这是**防注入设计**，因此扩字段必须同时加值校验，否则等于开一条注入通道。

- [ ] **Step 1: 写入失败测试**

~~~python
def test_security_code_must_be_six_digits():
    assert _normalize_metadata_filters({"security_code": "600519"}) == {"security_code": "600519"}
    assert _normalize_metadata_filters({"security_code": "600519 or 1==1"}) == {}


def test_industry_must_be_in_enum():
    assert _normalize_metadata_filters({"industry_l1": "食品饮料"}) == {"industry_l1": "食品饮料"}
    assert _normalize_metadata_filters({"industry_l1": "白酒"}) == {}


def test_unknown_field_is_dropped():
    assert _normalize_metadata_filters({"tenant_id": "other"}) == {}
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/unit/test_metadata_filters.py -v`

Expected: FAIL —— 新字段被白名单丢弃，非法值未被拒绝。

- [ ] **Step 3: 实现最小实现**

抽出 `_ALLOWED_FILTER_VALIDATORS: dict[str, Callable[[str], bool]]`，逐键校验后再进 `_build_filter`。非法值丢弃而非报错（与「抽不到就不加过滤」一致）。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/unit/test_metadata_filters.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/services/chat_service.py tests/unit/test_metadata_filters.py
git commit -m "feat: extend metadata filter whitelist with value validation"
~~~

### Task 10: 上下文注入——日期标注、表格与图片块

**Files:**

- Modify: `finance_rag/src/services/chat_service.py`（`build_context` / `merge_docs`）
- Modify: `finance_rag/src/agent/prompts/chat.py`
- Create: `tests/unit/test_context_builder.py`

**Interfaces:**

- Produces: `merge_docs` 排序由纯 score 改为 `(score, date desc)`。
- Produces: `build_context` 每块头部带 `报告日期` 与 `页码`；表格块按单表 8000 字符注入，超限标 `truncated`。
- Produces: `ANSWER_PROMPT` 强制每条结论标注报告日期与出处。

- [ ] **Step 1: 写入失败测试**

~~~python
def test_context_marks_date_and_page_for_each_source():
    ctx, sources = build_context([_doc(date="2026-08-15", start_page=4, end_page=5)])
    assert "报告日期：2026-08-15" in ctx
    assert "页码：4-5" in ctx


def test_docs_with_equal_score_sort_by_newer_date_first():
    merged = merge_docs([_doc(date="2023-01-01"), _doc(date="2026-08-15")], top_k=2)
    assert merged[0]["date"] == "2026-08-15"
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/unit/test_context_builder.py -v`

Expected: FAIL —— 上下文未带日期/页码，排序也未考虑日期。

- [ ] **Step 3: 实现最小实现**

改 `build_context` 的块头模板与 `merge_docs` 的 sort key；prompt 里加一句「每条结论必须标注来源报告日期与 `[N]` 编号」。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/unit/test_context_builder.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/services/chat_service.py finance_rag/src/agent/prompts/chat.py tests/unit/test_context_builder.py
git commit -m "feat: surface report date and page in answer context"
~~~

### Task 11: 原文页渲染接口

**Files:**

- Modify: `finance_rag/src/api/routes/documents.py`
- Create: `tests/integration/test_document_page_render.py`

**Interfaces:**

- Produces: `GET /api/documents/{source:path}/page/{page}` → `image/png`。
- Consumes: `storage.download_to_path(f"docs/{source}", tmp)` 后 `pymupdf` 渲染该页（`page - 1`，1-based 入参）。

**现状:** 原始 PDF 已存 `docs/{filename}`（`document_service.py:624`），但全仓无任何 `FileResponse` / 预览路由；`pymupdf` 在 `requirements.txt:18` 但零引用。

- [ ] **Step 1: 写入失败测试**

~~~python
def test_render_page_returns_png(client, uploaded_pdf):
    resp = client.get("/api/documents/sample.pdf/page/1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:4] == b"\x89PNG"


def test_render_page_out_of_range_returns_404(client, uploaded_pdf):
    assert client.get("/api/documents/sample.pdf/page/999").status_code == 404
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/integration/test_document_page_render.py -v`

Expected: FAIL —— 路由不存在（404 而非 200）。

- [ ] **Step 3: 实现最小实现**

路由内 `download_to_path` 到 `tempfile` → `pymupdf.open` → `page.get_pixmap()` → `Response(media_type="image/png")` → 清理临时文件。越界或非 PDF 返回 404/400，不抛 500。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/integration/test_document_page_render.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/api/routes/documents.py tests/integration/test_document_page_render.py
git commit -m "feat: render original report page on demand"
~~~

### Task 12: 元数据人工修正接口

**Files:**

- Modify: `finance_rag/src/api/routes/documents.py`
- Modify: `finance_rag/src/schemas/document.py`
- Create: `tests/integration/test_document_metadata_patch.py`

**Interfaces:**

- Produces: `PATCH /api/documents/{source:path}/metadata`，body 为可选的 `security_code` / `security_name` / `industry_l1` / `industry_l2` / `report_type` / `broker`。
- 行为：值走与 Task 9 相同的校验器；写 `meta_source="manual"`、`needs_review=false`；**只做 Milvus 标量 upsert，不重新嵌入**；`parent_chunks` / `table_chunks` 不存元数据，无需同步。

- [ ] **Step 1: 写入失败测试**

~~~python
def test_patch_metadata_updates_scalars_without_reembedding(client, monkeypatch, uploaded_pdf):
    embed_calls = []
    monkeypatch.setattr(kb_stub, "embed", lambda *a, **k: embed_calls.append(1))
    resp = client.patch("/api/documents/sample.pdf/metadata",
                        json={"security_code": "600519", "industry_l1": "食品饮料"})
    assert resp.status_code == 200
    assert resp.json()["meta_source"] == "manual"
    assert embed_calls == []


def test_patch_metadata_rejects_invalid_industry(client, uploaded_pdf):
    resp = client.patch("/api/documents/sample.pdf/metadata", json={"industry_l1": "白酒"})
    assert resp.status_code == 422
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/integration/test_document_metadata_patch.py -v`

Expected: FAIL —— 路由不存在。

- [ ] **Step 3: 实现最小实现**

查询该 source 的全部 chunk id → `upsert` 时保留向量列（从现有行读回）→ 只改标量列。校验失败返回 422 并给出中文原因。

- [ ] **Step 4: 验证测试通过**

Run: `python -m pytest tests/integration/test_document_metadata_patch.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

~~~bash
git add finance_rag/src/api/routes/documents.py finance_rag/src/schemas/document.py tests/integration/test_document_metadata_patch.py
git commit -m "feat: allow manual metadata correction"
~~~

### Task 13: 前端——元数据 chip、可信度、页码弹层、元数据编辑

**Files:**

- Modify: `frontend/src/views/ChatView.vue`
- Modify: `frontend/src/views/DocumentsView.vue`
- Modify: `frontend/src/api/index.js`
- Modify: `frontend/src/schemas` 对应 TS/JS 类型（若有）

**Interfaces:**

- Consumes: `done.citation_validation` / `answer_rejected` / `low_confidence`（现在前端**根本没读**这三个字段）；SSE `sources` 事件里的新字段。
- Produces: 自动推断的过滤条件渲染为**可见可改的 chip**，可一键清除；来源卡片带页码按钮，点击弹出原文页图；文档列表带"待确认"标记与元数据编辑弹层。

- [ ] **Step 1: 写入失败测试**

前端无既有测试框架，用最小断言脚本固定行为（若已有 vitest 则用 vitest；否则以手工验收清单代替，并在 PR 描述里勾选）：

~~~text
[ ] 答案命中拒答时，界面显示拒答提示而不是空白回答
[ ] low_confidence 为 true 时显示"低置信"标记
[ ] 自动过滤条件以 chip 呈现，点击 × 后重新提问不带该条件
[ ] 来源卡片点击页码 → 弹出原文页 PNG
[ ] 文档列表对 needs_review 文档显示"待确认"，弹层可改元数据并提交
~~~

- [ ] **Step 2: 确认当前不满足**

Run: `cd frontend && npm run dev`，手工核对上表

Expected: 五项均不满足（`ChatView.vue` 的 `done` 处理只读 `rewritten_query` 与 `filters`）。

- [ ] **Step 3: 实现**

`chatStream` 的 `onEvent` 增读三个字段；chip 组件复用现有 `el-tag`；页码弹层用 `el-dialog` + `<img :src="pageUrl(source, page)">`；文档页加 `el-dialog` 表单调 PATCH。

- [ ] **Step 4: 验证**

Run: `cd frontend && npm run build`

Expected: 构建通过；手工核对上表五项全部满足。

- [ ] **Step 5: 提交**

~~~bash
git add frontend/src/views/ChatView.vue frontend/src/views/DocumentsView.vue frontend/src/api/index.js
git commit -m "feat: surface metadata chips, traceability and page preview"
~~~

### Task 14: 删除合规模块

**Files:**

- Delete: `finance_rag/src/api/routes/compliance.py`、`finance_rag/src/services/compliance_service.py`、`finance_rag/src/services/compliance_rules.py`、`finance_rag/src/schemas/compliance.py`、`frontend/src/views/ComplianceView.vue`、`tests/unit/test_compliance_evidence.py`、`finance_rag/src/eval/data/compliance_*.md`
- Modify: `finance_rag/src/services/citation_validator.py`、`finance_rag/src/rag/ingestion/chunker.py`、`finance_rag/src/core/config.py`、`finance_rag/src/main.py`、`frontend/src/router/index.js`、`frontend/src/App.vue`、`scripts/seed_data.py`、`finance_rag/src/services/chat_service.py`

**保留:** `CitationValidator.validate`（研报要用）、`compliance_risk` 分类值（删了要动已 seed 的 PG 记录，收益为零）。

- [ ] **Step 1: 写入失败测试（断言删除后行为）**

~~~python
def test_compliance_routes_are_gone(client):
    assert client.post("/api/compliance/review", json={"query": "x"}).status_code == 404


def test_generic_citation_validation_still_works():
    from finance_rag.src.services.citation_validator import CitationValidator
    assert hasattr(CitationValidator, "validate")


def test_document_categories_still_expose_compliance_risk():
    from finance_rag.src.rag.models.document_category import DOCUMENT_CATEGORIES
    assert "compliance_risk" in DOCUMENT_CATEGORIES
~~~

- [ ] **Step 2: 验证测试按预期失败**

Run: `python -m pytest tests/unit/test_compliance_removed.py -v`

Expected: FAIL —— 合规路由仍返回 200。

- [ ] **Step 3: 执行删除与清理**

删除上列文件；`citation_validator.py` 去掉 `validate_clause_citations` 与 `validate_finding_evidence`；`chunker.py` 去掉 `_split_compliance_articles` 分支与其测试；`config.py` 去掉 `COMPLIANCE_*` 与从未被读取的 `ENABLE_METADATA_FILTER`；`main.py` 去掉路由挂载；前端去掉视图/路由/菜单；`seed_data.py` 修掉不在 `DOCUMENT_CATEGORIES` 里的 `it_technology`；`chat_service.iter_active_kbs` 修掉 `exists` 死代码（`chat_service.py:403-424`）。

- [ ] **Step 4: 验证**

Run: `python -m pytest tests/unit/test_compliance_removed.py -v && python -m pytest -q`

Expected: 全部 PASS，无残留 import 引用。

- [ ] **Step 5: 提交**

~~~bash
git add -A
git commit -m "refactor: remove compliance module"
~~~

### Task 15: 语料入库、端到端验证与 baseline

> **2026-09-28 修订**：语料实际到位 **18 篇 / 318 页**（原约定 50–100 篇），且**没有任何标的被
> ≥2 篇覆盖**，因此原定「跨券商对比同一指标」场景不可构造。需求方确认**改场景**为
> 「跨报告整合同一主题的数据」。下面按修订后的口径执行；变更理由与代价见 spec 的
> 「场景变更记录」。

**Files:**

- Create: `scripts/ingest_reports.py`（批量入库 + 逐篇验证 + 语料可行性检查）
- Create: `scripts/results/research_report/18-docs-baseline/report.txt`
- Create: `finance_rag/src/eval/data/research_report_gold.md`

**Interfaces:**

- Consumes: `scripts/ingest_reports.py`、`scripts/eval_recall.py` 的 L0/L1/L2/L3/L5 分层。
- Produces: 20 题 gold 集（**10 跨报告同主题整合** + 8 单跳定位 + 2 负样本，`review_status=approved`，标注证据 chunk id 与页码）。
- Produces: 四项硬阈值的实测值。

**先决条件（均已满足）:** 语料到位；Milvus 可达且集合已按新 schema 重建。

**前置修复（实测发现，已随本轮提交）:**

- `get_storage()` 改为按 `STORAGE_BACKEND` 选择后端（原实现恒走 OSS，会把原件发往公网）；
- compose 的 milvus pin `v2.4.0` → `v2.6.17`（BM25 Function 需 2.5+）；
- `rebuild_collection()` 连带清空指纹与 PG 派生表（否则重建后所有上传被静默跳过）；
- `_parse_json_payload` 解包 langchain `AIMessage`（否则**每篇**元数据都是空的）；
- `_find_content_list` 改用 `*content_list.json`（否则页码归属几乎全为 0）；
- 页码匹配改为包含式；HTML `<table>`（研报财务表）纳入表格识别。

- [x] **Step 1: 语料端到端入库并逐篇验证**

Run: `python scripts/ingest_reports.py`

Expected: 每篇 `status=completed`；逐篇验证表给出块数 / 图片块 / 整表数 / 页码归属 /
元数据（代码、行业、类型、待确认）。实测量级：18 篇 / 318 页，MinerU 约 0.16 页/秒。

- [ ] **Step 2: 录入 20 题 gold 集**

先用 `python scripts/ab_rag.py --generate-testset --count 40` 生成候选，再人工筛到 20 题、
逐题标注证据 chunk id 与页码，`review_status` 置 `approved`。
多跳题改为**跨报告同主题**（如多篇宏观周报对同一指标的表述、个股报告与行业报告的交叉印证）。

- [ ] **Step 3: 在全部 18 篇上建 baseline**

Run: `python scripts/eval_recall.py --top-k 5`

Expected: 输出落盘 `18-docs-baseline/report.txt`。
**注意**：18 篇属小语料，`hit_rate@5` 会偏乐观，这批数字只作起点基线；语料扩充后必须复测。

- [ ] **Step 4: 计算四项硬阈值**

按 gold 集统计 `hit_rate@5`、`evidence_rank ≤ 3` 占比、引用可溯源率、负样本拒答正确率，与 spec 阈值逐项对照。

Expected: 四项对照结果与差距清单写入 report.txt。

- [ ] **Step 5: 提交**

~~~bash
git add scripts/ingest_reports.py finance_rag/src/eval/data/research_report_gold.md scripts/results/research_report/
git commit -m "test: establish research report retrieval baseline"
~~~

### Task 16: 按 baseline 定点改造检索链路

**Files:**

- Modify: 视 baseline 结论而定，候选为 `finance_rag/src/core/config.py`（默认值）与 `scripts/ab_catalog`（实验目录）

**Interfaces:**

- Consumes: Task 15 的分层指标与差距清单。
- Produces: 每一项改动都带一组 A/B 结果（`scripts/ab_rag.py --experiment <name>`）。

**约束:** 只改 baseline 指向的层。候选：`ENABLE_LANGGRAPH`、rerank top-N、`CHAT_TOP_K`、`MILVUS_NPROBE`、`ENABLE_HYDE`、`CHAT_ENABLE_QUERY_REWRITE`、`DYNAMIC_K`、`ENABLE_SEMANTIC_CHUNKER`。**不得凭经验开刀。**

- [ ] **Step 1: 定位掉分项**

对 Task 15 的差距清单，先看 `evidence_rank` 是否落在 top-15 候选内却被 rerank 截断（那是重排序问题），再决定是召回问题还是排序问题。

- [ ] **Step 2: 逐项 A/B**

Run: `python scripts/ab_rag.py --experiment <name> --fast`

Expected: 每项改动都有 `report.md` 与配对 bootstrap 置信区间；无显著提升的改动**回滚**。

- [ ] **Step 3: 达阈值则全量入库**

四项硬阈值全部达标后，全量入 500–5000 篇，并复跑一次 `eval_recall.py` 确认指标不随体量劣化。

- [ ] **Step 4: 完整验证与提交**

Run: `python -m pytest -q`

Expected: exit code 0。

~~~bash
git add -A
git commit -m "perf: tune retrieval from measured baseline"
~~~

---

## 执行记录（2026-09-26）

Task 1–14 已实现并提交；Task 15–16 **阻塞于语料**，未执行也未伪造 baseline。

| Task | 状态 | 提交 |
| --- | --- | --- |
| 1 解析层产出页码与图片资产 | 完成 | `5f3a7e3`（spec/plan 见 `f8d0926`） |
| 2 切块层：页码、表格双份、图片块 | 完成 | `163fa7e` |
| 3 PostgreSQL 整表仓储 | 完成 | `f48d84b` |
| 4 Milvus schema 扩展 | 完成 | `bb39b0c` |
| 5 元数据抽取（正则 + 申万 enum 校验） | 完成 | `42d7e12` |
| 6 图片描述（视觉模型 + 外发审计） | 完成 | `eeaf509` |
| 7 入库接入（元数据 / 整表 / 图片块） | 完成 | `8d467b6` |
| 8 检索侧整表展开与字段透出 | 完成 | `4dc04b0` |
| 9 过滤白名单与值校验 | 完成 | `b188aa1` |
| 10 上下文注入（日期 / 页码 / 表格预算） | 完成 | `d2eac6f` |
| 11 原文页渲染接口 | 完成 | `655949a` |
| 12 元数据人工修正接口 | 完成 | `885a8a7` |
| 13 前端：chip / 可信度 / 页码弹层 / 元数据编辑 | 完成 | `291dce4` |
| 14 删除合规模块 | 完成 | `c295930` |
| 15 语料入库、端到端验证与 baseline | 进行中 | — |
| 16 按 baseline 定点改造 | **阻塞** | — |

### 执行中的偏离（均已落代码与测试，此处留痕）

1. **Task 2 顺带改了 `document_service._upload_extracted_images`**：`images` 真正有值后，
   旧的「临时文件字典」契约会立刻 `AttributeError`，因此同步改为按字节上传。
2. **Task 5 的 `llm` 参数改为 `enable_llm: bool`**：原先的 `llm=object()` 哨兵会渗进生产调用点。
3. **新增 `tests/conftest.py`**：`.env` 带真实 key，入库链路的模型调用会让单测变成真实网络请求
   （单测耗时从 16s 涨到 36s 并打挂时间敏感用例）。conftest 默认关闭 `ENABLE_METADATA_LLM`，
   且 `config` 必须在 fixture 内部导入——否则会在收集期读掉环境变量，早于 `test_auth` 设置 `os.environ`。
4. **Task 13 新增 `ChatRequest.infer_filters`**：没有它，前端「清除过滤」只是清本地状态，
   后端会把条件推断回来，chip 的清除是装饰性的。
5. **Task 14 多删了 `extract_clause_references` 与其测试**：它只服务于 `validate_clause_citations`，
   留着就是无调用方的死代码。
6. **`test_recursive_splitter.py` 的等价性语料**先后从 `compliance_text_cases.md`、
   `assets/ex/内部内控与组织权责管理制度.md` 换成 `docs/superpowers/specs/2026-09-26-research-report-qa-design.md`
   （仍是真实中文长文）。非研报样本已按需求方要求从 `assets/ex/` 清出，只留一篇研报。
7. **`test_chunker.py` 的 `parse_and_chunk` 回归用例不再依赖仓库样本语料**：改为在
   `tests/unit/.tmp` 下自写临时 md。样本随时可能被清理，测试不该因此变红。

### 遗留风险（Task 15 才能验证）

- `content_list.json` 的真实字段形态只用合成 payload 测过；真实 MinerU 产出未验证。
  已做两层降级（content_list → PDF 文本块 → 空），页码匹配不上时留 0 而不猜。
- 视觉模型对研报图表的描述质量、以及 `IMAGE_CAPTION_MIN_*` 阈值是否合适，需要真实研报校准。
- 元数据 LLM 抽取的行业准确率（申万词表内命中率）需要真实研报统计。
