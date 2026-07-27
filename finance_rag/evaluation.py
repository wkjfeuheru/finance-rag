"""策略评估模块（基于 ragas 框架）。

使用 ragas 对不同检索策略（稠密/稀疏权重、是否启用重排序）进行评估，
输出 faithfulness、answer_relevancy、context_precision、context_recall 等指标。

核心组件：
* :class:`StrategyConfig` — 策略配置（权重、重排序开关、检索深度）
* :class:`TestSetEntry` — 测试集条目（查询 + 标准答案 + 元数据）
* :class:`TestSetLoader` — 从 files/docs/evaluation_qa.md 解析加载静态测试集
* :class:`StrategyEvaluator` — ragas 策略评估器

标准答案已固化在 ``files/docs/evaluation_qa.md`` 中，运行评估时不会调用大模型生成答案。
"""

from __future__ import annotations

import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 将项目级 libs 目录加入 sys.path（ragas 及其依赖安装在此）
_LIBS_DIR = Path(__file__).resolve().parents[1] / "libs"
if _LIBS_DIR.exists() and str(_LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(_LIBS_DIR))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ragas.llms.base import LangchainLLMWrapper
from ragas.prompt.utils import extract_json

from .config import (
    CHAT_RERANK_TOP_K,
    CHAT_TOP_K,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    RAGAS_MAX_RETRIES,
    RAGAS_MAX_WORKERS,
    RAGAS_TIMEOUT_SECONDS,
    model,
)
from .knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


class JsonExtractLangchainLLMWrapper(LangchainLLMWrapper):
    """Extract the JSON payload from Ragas judge-model responses.

    DeepSeek may wrap JSON in Markdown or explanatory text.  Ragas validates
    judge responses against Pydantic models, so normalize every generated
    candidate before the prompt parser sees it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_outputs: list[str] = []

    def clear_captured_outputs(self) -> None:
        self.captured_outputs.clear()

    async def agenerate_text(
        self,
        prompt,
        n=1,
        temperature=0.01,
        stop=None,
        callbacks=None,
    ):
        result = await super().agenerate_text(
            prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )
        for batch in result.generations:
            for generation in batch:
                if generation.text:
                    raw_text = generation.text
                    self.captured_outputs.append(raw_text)
                    generation.text = extract_json(raw_text)
        return result


def _json_safe_metric(value: Any) -> float | None:
    """Convert a metric value to strict-JSON-compatible form.

    Ragas may return NaN or infinity when a metric cannot be calculated for a
    sample.  Starlette deliberately rejects those values during JSON encoding,
    so represent unavailable metrics as JSON ``null`` instead.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, 4) if math.isfinite(numeric) else None


def _text_units(text: str) -> set[str]:
    """Return stable Chinese bigrams and alphanumeric words for overlap scoring."""
    normalized = re.sub(r"\[\d+\]", " ", (text or "").lower())
    units = set(re.findall(r"[a-z0-9]+", normalized))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(sequence) == 1:
            units.add(sequence)
        else:
            units.update(sequence[i:i + 2] for i in range(len(sequence) - 1))
    return units


def _token_f1(left: str, right: str) -> float:
    left_units, right_units = _text_units(left), _text_units(right)
    if not left_units or not right_units:
        return 0.0
    common = len(left_units & right_units)
    if common == 0:
        return 0.0
    precision = common / len(left_units)
    recall = common / len(right_units)
    return 2 * precision * recall / (precision + recall)


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[。！？!?；;\n]+", text or "")
        if len(part.strip()) >= 4
    ]


def _supported_sentence_score(claims: str, evidence: list[str]) -> float:
    """Average support of each claim sentence by its best evidence chunk."""
    claim_list = _sentences(claims)
    evidence_list = [item for item in evidence if item]
    if not claim_list or not evidence_list:
        return 0.0
    scores = [
        max((_token_f1(claim, item) for item in evidence_list), default=0.0)
        for claim in claim_list
    ]
    return round(sum(scores) / len(scores), 4)


def _fallback_metrics(sample: dict[str, Any]) -> dict[str, float]:
    """Deterministic local fallbacks for LLM-judge metrics."""
    response = sample.get("response", "")
    contexts = sample.get("retrieved_contexts", []) or []
    return {
        "faithfulness": _supported_sentence_score(response, contexts),
    }

# 静态测试集文件路径（标准答案已固化在此 MD 文件中）
_TEST_SET_FILE = Path(__file__).resolve().parents[1] / "files" / "docs" / "evaluation_qa.md"
# 生成回答的 prompt
_ANSWER_PROMPT = """你是一个专业的金融知识助手。请基于以下检索到的金融文档内容回答用户问题。

回答要求：
1. 严格基于【检索内容】回答，不要编造未提供的信息
2. 在引用某段内容时，标注引用编号，如 [1]、[2]
3. 如果检索内容不足以回答问题，请明确说明"根据现有知识库内容，暂无法完整回答该问题"
4. 只回答问题直接询问的内容，不补充检索内容未明确陈述的原因、背景、结论或外部知识
5. 使用简洁中文，优先复述检索内容中的明确事实，避免不必要的扩展
6. 涉及具体数字、条件、操作步骤时，务必保持与检索内容一致并引用对应来源

【检索内容】
{context}

【用户问题】
{query}

请开始回答："""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """检索策略配置。

    Attributes:
        dense_weight: 稠密向量权重（0.0-1.0）。
        sparse_weight: 稀疏向量权重（0.0-1.0）。
        use_rerank: 是否启用 BGE 重排序。
        rerank_top_n: 重排序返回的结果数。
        k: 检索深度（返回的上下文数）。
    """

    dense_weight: float = HYBRID_DENSE_WEIGHT
    sparse_weight: float = HYBRID_SPARSE_WEIGHT
    use_rerank: bool = False
    rerank_top_n: int = CHAT_RERANK_TOP_K
    k: int = CHAT_TOP_K

    def to_dict(self) -> dict[str, Any]:
        return {
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "use_rerank": self.use_rerank,
            "rerank_top_n": self.rerank_top_n,
            "k": self.k,
        }

    @property
    def label(self) -> str:
        """策略简短标签，用于结果展示。"""
        return f"dense={self.dense_weight:.1f}|rerank={'on' if self.use_rerank else 'off'}"


@dataclass
class TestSetEntry:
    """测试集条目。

    Attributes:
        query: 测试查询文本。
        ground_truth: 标准答案（来自 files/docs/evaluation_qa.md）。
        related_docs: 相关文档列表。
    """

    query: str
    question_id: int = 0
    ground_truth: str = ""
    related_docs: str = ""
    related_sections: str = ""
    required_facts: tuple[str, ...] = ()
    test_type: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "related_docs": self.related_docs,
            "related_sections": self.related_sections,
            "required_facts": list(self.required_facts),
            "test_type": self.test_type,
        }


# ---------------------------------------------------------------------------
# 测试集加载器（从静态 MD 文件解析）
# ---------------------------------------------------------------------------

class TestSetLoader:
    """从 ``files/docs/evaluation_qa.md`` 解析加载静态测试集。

    MD 文件结构：每个问题以 ``## N. 问题`` 开头，后接相关文档，
    再以 ``### 标准答案`` 引出答案内容，以 ``---`` 分隔条目。
    """

    # 匹配问题标题：## 1. 问题内容
    _QUESTION_PATTERN = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$", re.MULTILINE)
    _RELATED_DOCS_PATTERN = re.compile(r"^-\s*相关文档：\s*(.+?)\s*$", re.MULTILINE)
    _RELATED_SECTIONS_PATTERN = re.compile(r"^-\s*相关章节：\s*(.+?)\s*$", re.MULTILINE)
    _REQUIRED_FACTS_PATTERN = re.compile(r"^-\s*必需事实：\s*(.+?)\s*$", re.MULTILINE)
    _TEST_TYPE_PATTERN = re.compile(r"^-\s*测试类型：\s*(.+?)\s*$", re.MULTILINE)
    def __init__(self, file_path: Path | None = None):
        self._file_path = file_path or _TEST_SET_FILE

    def load_test_set(self) -> list[TestSetEntry]:
        """从 MD 文件解析测试集。

        Returns:
            测试集条目列表；文件不存在时返回空列表。
        """
        if not self._file_path.exists():
            logger.warning("测试集文件不存在：%s", self._file_path)
            return []

        text = self._file_path.read_text(encoding="utf-8")
        entries = self._parse(text)
        self._validate(entries)
        return entries

    def has_ground_truth(self) -> bool:
        """检查测试集是否包含标准答案。"""
        entries = self.load_test_set()
        return bool(entries) and all(e.ground_truth for e in entries)

    def _parse(self, text: str) -> list[TestSetEntry]:
        """解析 MD 文本为测试集条目列表。"""
        entries: list[TestSetEntry] = []
        matches = list(self._QUESTION_PATTERN.finditer(text))

        for i, match in enumerate(matches):
            question_id = int(match.group(1))
            query = match.group(2).strip()
            # 当前问题到下一个问题（或文档末尾）的文本块
            block_start = match.end()
            block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[block_start:block_end]

            # 提取标准答案（### 标准答案 到下一个 --- 之间）
            ground_truth = self._extract_ground_truth(block)
            related_docs_match = self._RELATED_DOCS_PATTERN.search(block)
            related_docs = related_docs_match.group(1).strip() if related_docs_match else ""
            sections_match = self._RELATED_SECTIONS_PATTERN.search(block)
            related_sections = sections_match.group(1).strip() if sections_match else ""
            facts_match = self._REQUIRED_FACTS_PATTERN.search(block)
            required_facts = tuple(
                fact.strip()
                for fact in re.split(r"[；;|]", facts_match.group(1))
                if fact.strip()
            ) if facts_match else ()
            type_match = self._TEST_TYPE_PATTERN.search(block)
            test_type = type_match.group(1).strip() if type_match else "standard"

            entries.append(TestSetEntry(
                question_id=question_id,
                query=query,
                ground_truth=ground_truth,
                related_docs=related_docs,
                related_sections=related_sections,
                required_facts=required_facts,
                test_type=test_type,
            ))

        logger.info("从 %s 解析到 %d 条测试查询", self._file_path, len(entries))
        return entries

    @staticmethod
    def _validate(entries: list[TestSetEntry]) -> None:
        """校验问题和标准答案。"""
        errors: list[str] = []
        seen_queries: set[str] = set()
        for index, entry in enumerate(entries, 1):
            if not entry.query:
                errors.append(f"第 {index} 条缺少问题")
            elif entry.query in seen_queries:
                errors.append(f"第 {index} 条问题重复：{entry.query}")
            seen_queries.add(entry.query)

            if not entry.ground_truth:
                errors.append(f"第 {index} 条缺少标准答案")
            if not entry.related_docs:
                errors.append(f"第 {index} 条缺少相关文档")
            elif any(Path(item.strip()).name != item.strip()
                     for item in re.split(r"[,，;；|]", entry.related_docs) if item.strip()):
                errors.append(f"第 {index} 条相关文档路径无效：{entry.related_docs}")

        if errors:
            raise ValueError("评估测试集校验失败：" + "；".join(errors))

    @staticmethod
    def _extract_ground_truth(block: str) -> str:
        """从问题块中提取标准答案文本。

        答案位于 ``### 标准答案`` 之后，到下一个 ``---`` 分隔符或块末尾。
        """
        marker = "### 标准答案"
        idx = block.find(marker)
        if idx == -1:
            return ""

        # 跳过 marker 行
        rest = block[idx + len(marker):]
        # 截断到下一个分隔符 ---
        sep_idx = rest.find("\n---")
        if sep_idx != -1:
            rest = rest[:sep_idx]

        # 去除首尾空白行
        return rest.strip()


# ---------------------------------------------------------------------------
# 策略评估器
# ---------------------------------------------------------------------------

class StrategyEvaluator:
    """使用 ragas 框架评估检索策略。

    评估流程：
    1. 对每条测试查询执行混合检索（按策略配置）
    2. 使用 LLM 基于检索结果生成回答
    3. 构建 ragas EvaluationDataset
    4. 调用 ragas.evaluate 计算指标

    支持的评估指标：
    * faithfulness — 回答对检索内容的忠实度
    * answer_relevancy — 回答与问题的相关性
    * context_precision — 检索内容的精确度
    * context_recall — 检索内容的召回率
    """

    def __init__(self):
        self._kb = get_knowledge_base()

    def evaluate(
        self,
        config: StrategyConfig,
        entry: TestSetEntry,
    ) -> dict[str, Any]:
        """评估单个策略配置下的单条查询。

        Args:
            config: 策略配置（权重、重排序等）。
            entry: 测试集条目（必须含 ground_truth）。

        Returns:
            评估结果字典，含单条指标、检索上下文、生成回答和标准答案。
        """
        if model is None:
            raise ValueError("LLM 未配置（缺少 DEEPSEEK_API_KEY）")

        if not entry.ground_truth:
            raise ValueError(
                "该查询缺少标准答案，请检查 files/docs/evaluation_qa.md 文件"
            )

        query = entry.query
        logger.info("评估 [%s]：%s", config.label, query[:50])

        # 1. 检索
        try:
            docs = self._kb.hybrid_search(
                query,
                k=config.k,
                dense_weight=config.dense_weight,
                sparse_weight=config.sparse_weight,
                expand_parents=True,
                use_rerank=config.use_rerank,
                rerank_top_n=config.rerank_top_n,
            )
        except Exception as exc:
            logger.error("检索失败 [%s]：%s", query[:30], exc)
            docs = []

        contexts = [d.get("content", "") for d in docs if d.get("content")]
        sources = [
            {
                "title": d.get("title", ""),
                "source": d.get("source", ""),
                "score": d.get("score", 0.0),
                "content": d.get("content", "")[:500],
            }
            for d in docs
        ]

        # 2. 生成回答
        context_text = self._build_context(contexts)
        answer_prompt = ChatPromptTemplate.from_messages([("human", _ANSWER_PROMPT)])
        answer_chain = answer_prompt | model | StrOutputParser()

        try:
            response = answer_chain.invoke({
                "context": context_text,
                "query": query,
            }).strip()
        except Exception as exc:
            logger.error("生成回答失败 [%s]：%s", query[:30], exc)
            response = f"生成回答失败：{exc}"

        # 3. 构建 ragas 数据集（单条）并评估
        ragas_data = [{
            "user_input": query,
            "retrieved_contexts": contexts,
            "response": response,
            "reference": entry.ground_truth,
        }]

        return self._run_ragas(ragas_data, config, entry, sources, response)

    def _build_context(self, contexts: list[str]) -> str:
        """将检索结果组装为 LLM 上下文文本。"""
        if not contexts:
            return "（未检索到相关文档）"
        chunks: list[str] = []
        for i, content in enumerate(contexts, 1):
            truncated = content[:2000] if content else ""
            chunks.append(f"[{i}] {truncated}")
        return "\n\n---\n\n".join(chunks)

    def _run_ragas(
        self,
        ragas_data: list[dict[str, Any]],
        config: StrategyConfig,
        entry: TestSetEntry | None = None,
        sources: list[dict[str, Any]] | None = None,
        response: str = "",
    ) -> dict[str, Any]:
        """调用 ragas.evaluate 计算评估指标。"""
        from ragas import EvaluationDataset, evaluate
        from ragas.run_config import RunConfig

        # 构建评估数据集
        eval_dataset = EvaluationDataset.from_list(ragas_data)

        run_config = RunConfig(
            timeout=RAGAS_TIMEOUT_SECONDS,
            max_retries=RAGAS_MAX_RETRIES,
            max_workers=RAGAS_MAX_WORKERS,
        )

        # 包装 LLM
        evaluator_llm = JsonExtractLangchainLLMWrapper(
            model,
            run_config=run_config,
        )

        # 显式注入评估 LLM 和项目使用的 DashScope embeddings。
        # 如果不传 embeddings，Ragas 会回退到 OpenAI embeddings，导致错误
        # 使用无额度/未配置的 OPENAI_API_KEY。
        from ragas.embeddings import LangchainEmbeddingsWrapper

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            self._kb._get_embeddings()
        )
        metrics = self._get_metrics(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        logger.info("开始 ragas 评估，指标：%s", [m.name for m in metrics])

        metrics_dict: dict[str, float | None] = {}
        metric_errors: dict[str, str] = {}

        # Run all metrics together first to retain Ragas' parallel execution.
        # Failed metrics become NaN; only those metrics are retried separately
        # with exception propagation so their real failure reason is visible.
        initial_result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
            show_progress=False,
        )
        for metric in metrics:
            metric_name = metric.name
            try:
                raw_values = initial_result[metric_name]
            except (KeyError, TypeError):
                # Ragas may omit a metric column entirely when every sample
                # for that metric failed, instead of returning a NaN column.
                raw_values = []
            values = [
                score
                for score in (
                    _json_safe_metric(value)
                    for value in raw_values
                )
                if score is not None
            ]
            metrics_dict[metric_name] = (
                round(sum(values) / len(values), 4) if values else None
            )

        for metric in metrics:
            metric_name = metric.name
            if metrics_dict[metric_name] is not None:
                continue
            evaluator_llm.clear_captured_outputs()
            try:
                retry_result = evaluate(
                    dataset=eval_dataset,
                    metrics=[metric],
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    run_config=run_config,
                    raise_exceptions=True,
                    show_progress=False,
                )
                retry_values = [
                    score
                    for score in (
                        _json_safe_metric(value)
                        for value in retry_result[metric_name]
                    )
                    if score is not None
                ]
                if not retry_values:
                    raise ValueError("Ragas returned no finite score")
                metrics_dict[metric_name] = round(
                    sum(retry_values) / len(retry_values),
                    4,
                )
            except Exception as exc:
                metric_errors[metric_name] = str(exc)
                raw_output = (
                    evaluator_llm.captured_outputs[-1]
                    if evaluator_llm.captured_outputs
                    else "<no LLM output>"
                )
                logger.warning(
                    "Ragas metric %s retry failed: %s; last LLM output: %s",
                    metric_name,
                    exc,
                    raw_output[:2000],
                )

        per_query: list[dict[str, Any]] = []
        for sample in ragas_data:
            row_data: dict[str, Any] = {
                "query": sample.get("user_input", ""),
                "response": sample.get("response", ""),
            }
            row_data.update(metrics_dict)
            per_query.append(row_data)

        # Faithfulness retains a deterministic fallback. Context recall must
        # remain the native Ragas result and is never locally supplemented.
        fallback_metric_names: list[str] = []
        if ragas_data:
            local_scores = _fallback_metrics(ragas_data[0])
            for name, score in local_scores.items():
                if metrics_dict.get(name) is None:
                    metrics_dict[name] = score
                    fallback_metric_names.append(name)
        for row_data in per_query:
            row_data.update(metrics_dict)

        result_dict: dict[str, Any] = {
            "strategy": config.to_dict(),
            "metrics": metrics_dict,
            "fallback_metrics": fallback_metric_names,
            "metric_errors": metric_errors,
            "per_query": per_query,
            "query_count": len(ragas_data),
        }

        # 单条评估时附加详情
        if entry is not None:
            result_dict.update({
                "query": entry.query,
                "ground_truth": entry.ground_truth,
                "response": response,
                "sources": sources or [],
                "related_docs": entry.related_docs,
            })

        return result_dict

    @staticmethod
    def _get_metrics(llm=None, embeddings=None):
        """获取 ragas 评估指标列表。

        ragas 0.4 使用类形式指标，兼容实例形式。
        """
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            Faithfulness,
            LLMContextRecall,
        )

        metrics = [
            Faithfulness(llm=llm),
            # DeepSeek's LangChain wrapper returns one generation per request;
            # strictness=1 avoids requesting three generations and the related
            # partial-result warning.
            AnswerRelevancy(
                llm=llm,
                embeddings=embeddings,
                strictness=1,
            ),
            ContextPrecision(llm=llm),
            LLMContextRecall(llm=llm),
        ]

        return metrics


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_EVALUATOR_INSTANCE: StrategyEvaluator | None = None
_LOADER_INSTANCE: TestSetLoader | None = None


def get_strategy_evaluator() -> StrategyEvaluator:
    """获取全局 StrategyEvaluator 单例。"""
    global _EVALUATOR_INSTANCE
    if _EVALUATOR_INSTANCE is None:
        _EVALUATOR_INSTANCE = StrategyEvaluator()
    return _EVALUATOR_INSTANCE


def get_test_set_loader() -> TestSetLoader:
    """获取全局 TestSetLoader 单例。"""
    global _LOADER_INSTANCE
    if _LOADER_INSTANCE is None:
        _LOADER_INSTANCE = TestSetLoader()
    return _LOADER_INSTANCE

