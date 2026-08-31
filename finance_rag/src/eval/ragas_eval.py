"""策略评估模块（基于 ragas 框架）。

使用 ragas 对 RAG 优化点 A/B 实验的检索与回答质量进行评估，
输出 faithfulness、answer_relevancy、context_precision、context_recall、
answer_correctness、context_entity_recall 等指标。

核心组件：
* :class:`RagasBatchEvaluator` — 批量评估器：一次评估一臂的全部题目
  （打包为单个 EvaluationDataset，显著降低开销与指标抖动）；
* :func:`compute_composite_score` — 分层加权综合评分；
* 本地检索指标（hit_rate / MRR / NDCG）与对抗层拒绝分数。
"""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

# 将项目级 libs 目录加入 sys.path（ragas 及其依赖安装在此）
# 确保项目根目录在 sys.path 中：既支持 `python -m finance_rag.src.eval.ragas_eval`，
# 也支持直接 `python finance_rag/src/eval/ragas_eval.py` 运行
# （直接运行脚本时 sys.path[0] 是脚本目录，会报 ModuleNotFoundError: config）。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 将项目级 libs 目录加入 sys.path（ragas 及其依赖安装在此），优先于已安装版本
_LIBS_DIR = _PROJECT_ROOT / "libs"
if _LIBS_DIR.exists() and str(_LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(_LIBS_DIR))

# 兼容 langchain-community >= 0.4 — vertexai 模块已被移除，
# 但 ragas 0.4.3 在模块顶层导入了该路径。此处注入占位模块避免 ImportError。
try:
    from langchain_community.chat_models import vertexai  # noqa: F401
except ImportError:
    import types
    _dummy = types.ModuleType("langchain_community.chat_models.vertexai")
    _dummy.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[assignment]
    sys.modules["langchain_community.chat_models.vertexai"] = _dummy

from ragas.llms.base import LangchainLLMWrapper
from ragas.prompt.utils import extract_json

from finance_rag.src.core.config import (
    RAGAS_MAX_RETRIES,
    RAGAS_MAX_WORKERS,
    RAGAS_TIMEOUT_SECONDS,
    get_model,
)
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

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


# ---------------------------------------------------------------------------
# 本地检索指标
# ---------------------------------------------------------------------------

def _compute_local_retrieval_metrics(
    retrieved_titles: list[str],
    related_docs: str,
    k: int = 5,
) -> dict[str, float]:
    """基于检索到的文档标题和相关文档字段计算本地检索指标。

    Args:
        retrieved_titles: 检索返回的文档标题列表（按 rank 排序）。
        related_docs: 相关文档字段（逗号/分号分隔的文档名）。
        k: 检索深度。

    Returns:
        含 hit_rate@k, MRR@k, NDCG@k 的字典。
    """
    if not related_docs or not retrieved_titles:
        return {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}

    # 标准化：去除扩展名进行比较
    related_set = set()
    for doc in re.split(r"[,，;；|]", related_docs):
        doc = doc.strip()
        if doc:
            related_set.add(Path(doc).stem.lower())

    retrieved_stems = [Path(t).stem.lower() for t in retrieved_titles[:k]]

    # hit_rate@k: 至少有一个相关文档出现在 top-k 中
    hit = 1.0 if any(s in related_set for s in retrieved_stems) else 0.0

    # MRR@k: 第一个相关文档的倒数排名
    mrr = 0.0
    for rank, stem in enumerate(retrieved_stems, 1):
        if stem in related_set:
            mrr = 1.0 / rank
            break

    # NDCG@k: 简化版（相关性二值：相关=1，不相关=0）
    dcg = 0.0
    for rank, stem in enumerate(retrieved_stems, 1):
        if stem in related_set:
            dcg += 1.0 / (math.log2(rank + 1))
    # 理想 DCG（所有相关文档排在前面）
    ideal_count = min(len(related_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    ndcg = round(dcg / idcg, 4) if idcg > 0 else 0.0

    return {
        "hit_rate": round(hit, 4),
        "mrr": round(mrr, 4),
        "ndcg": ndcg,
    }


def build_context_text(contexts: list[str]) -> str:
    """将检索结果组装为 LLM 上下文文本。"""
    if not contexts:
        return "（未检索到相关文档）"
    chunks: list[str] = []
    for i, content in enumerate(contexts, 1):
        truncated = content[:2000] if content else ""
        chunks.append(f"[{i}] {truncated}")
    return "\n\n---\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Ragas 指标与批量评估
# ---------------------------------------------------------------------------

def _get_metrics(llm=None, embeddings=None, *, fast: bool = False):
    """获取 ragas 评估指标列表。

    Args:
        fast: True 时仅返回 3 项核心指标（faithfulness/answer_relevancy/
              context_precision），跳过 answer_correctness、context_recall、
              context_entity_recall。完整模式约 90-180s，快速模式约 30-60s。
    """
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextEntityRecall,
        ContextPrecision,
        Faithfulness,
        LLMContextRecall,
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
            strictness=1,
        ),
        ContextPrecision(llm=llm),
    ]

    if not fast:
        metrics.extend([
            LLMContextRecall(llm=llm),
            AnswerCorrectness(llm=llm, embeddings=embeddings),
            ContextEntityRecall(llm=llm),
        ])

    return metrics


class RagasBatchEvaluator:
    """Ragas 批量评估器：一次评估一臂的全部题目。

    ``evaluate_batch`` 将整臂样本打包为单个 ``EvaluationDataset`` 调用
    ``ragas.evaluate``（指标实例只初始化一次），再按样本拆分指标值，
    失败指标单独重试，faithfulness 保留确定性本地回退。
    """

    def __init__(self, kb=None):
        self._kb = kb or get_knowledge_base()

    def evaluate_batch(
        self,
        rows: list[dict[str, Any]],
        *,
        fast: bool = False,
    ) -> dict[str, Any]:
        """批量评估。

        Args:
            rows: 样本列表，每项必须含 ``user_input`` / ``retrieved_contexts`` /
                  ``response`` / ``reference``（可附其它自定义字段，会原样
                  回填到 per_query 输出中）。
            fast: True 时仅计算 3 项核心指标。

        Returns:
            ``{"metrics": {指标名: 均值}, "per_query": [{..., "metrics": {...}}],
            "metric_errors": {指标名: 错误信息}, "fallback_metrics": [...]}``
        """
        if get_model() is None:
            raise ValueError("LLM 未配置（缺少 DEEPSEEK_API_KEY）")
        if not rows:
            return {
                "metrics": {}, "per_query": [],
                "metric_errors": {}, "fallback_metrics": [],
            }

        from ragas import EvaluationDataset, evaluate
        from ragas.run_config import RunConfig
        from ragas.embeddings import LangchainEmbeddingsWrapper

        eval_dataset = EvaluationDataset.from_list([
            {
                "user_input": row["user_input"],
                "retrieved_contexts": row.get("retrieved_contexts") or [],
                "response": row.get("response") or "",
                "reference": row.get("reference") or "",
            }
            for row in rows
        ])

        run_config = RunConfig(
            timeout=RAGAS_TIMEOUT_SECONDS,
            max_retries=RAGAS_MAX_RETRIES,
            max_workers=RAGAS_MAX_WORKERS,
        )

        # 包装 LLM（DeepSeek JSON 兼容）与本地嵌入
        evaluator_llm = JsonExtractLangchainLLMWrapper(
            get_model(),
            run_config=run_config,
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            self._kb.get_embeddings()
        )
        metrics = _get_metrics(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            fast=fast,
        )
        logger.info("开始 ragas 批量评估（%d 样本），指标：%s",
                    len(rows), [m.name for m in metrics])

        # 全部指标一次性并行评估；失败指标单独重试
        initial_result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
            show_progress=False,
        )

        def _extract_metric_column(metric_name: str) -> list[float | None]:
            try:
                raw_values = initial_result[metric_name]
            except (KeyError, TypeError):
                return [None] * len(rows)
            values = [_json_safe_metric(v) for v in raw_values]
            return values

        columns: dict[str, list[float | None]] = {}
        metric_errors: dict[str, str] = {}
        for metric in metrics:
            columns[metric.name] = _extract_metric_column(metric.name)

        # 全部失败的指标单独重试（异常可见，便于定位真实原因）
        for metric in metrics:
            if any(value is not None for value in columns[metric.name]):
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
                columns[metric.name] = [
                    _json_safe_metric(v) for v in retry_result[metric.name]
                ]
            except Exception as exc:
                metric_errors[metric.name] = str(exc)
                raw_output = (
                    evaluator_llm.captured_outputs[-1]
                    if evaluator_llm.captured_outputs
                    else "<no LLM output>"
                )
                logger.warning(
                    "Ragas metric %s retry failed: %s; last LLM output: %s",
                    metric.name, exc, raw_output[:2000],
                )

        # faithfulness 确定性本地回退
        fallback_names: list[str] = []
        for i, row in enumerate(rows):
            if columns["faithfulness"][i] is None:
                columns["faithfulness"][i] = _fallback_metrics(row)[
                    "faithfulness"
                ]
                fallback_names.append("faithfulness")

        # 逐样本输出（保留调用方附加字段）
        per_query: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            item = {k: v for k, v in row.items() if k != "retrieved_contexts"}
            item["metrics"] = {
                name: columns[name][i] for name in columns
            }
            per_query.append(item)

        # 聚合均值（仅统计有限值）
        aggregated: dict[str, float | None] = {}
        for name, values in columns.items():
            finite = [v for v in values if v is not None]
            aggregated[name] = (
                round(sum(finite) / len(finite), 4) if finite else None
            )

        return {
            "metrics": aggregated,
            "per_query": per_query,
            "metric_errors": metric_errors,
            "fallback_metrics": sorted(set(fallback_names)),
        }


# ---------------------------------------------------------------------------
# 综合评分
# ---------------------------------------------------------------------------

# 默认权重（适用于 all 层级）
_DEFAULT_WEIGHTS = {
    "answer_correctness": 0.30,
    "faithfulness": 0.20,
    "answer_relevancy": 0.15,
    "context_recall": 0.15,
    "context_precision": 0.10,
    "hit_rate": 0.10,
}

# 分层权重
_TIER_WEIGHTS: dict[str, dict[str, float]] = {
    "L1": {
        "answer_correctness": 0.15,
        "faithfulness": 0.15,
        "answer_relevancy": 0.10,
        "context_recall": 0.10,
        "context_precision": 0.10,
        "hit_rate": 0.20,
        "mrr": 0.10,
        "ndcg": 0.10,
    },
    "L2": {
        "answer_correctness": 0.30,
        "faithfulness": 0.25,
        "answer_relevancy": 0.15,
        "context_recall": 0.15,
        "context_precision": 0.10,
        "hit_rate": 0.05,
    },
    "L3": {
        "negative_rejection": 0.40,
        "faithfulness": 0.20,
        "answer_correctness": 0.15,
        "answer_relevancy": 0.15,
        "context_precision": 0.10,
    },
}


def compute_composite_score(
    metrics: dict[str, float | None],
    tier: str = "all",
) -> dict[str, Any]:
    """根据指标值和测试层级计算加权综合分。"""
    weights = _TIER_WEIGHTS.get(tier, _DEFAULT_WEIGHTS)
    total_weight = 0.0
    weighted_sum = 0.0
    included: dict[str, float] = {}
    for name, weight in weights.items():
        value = metrics.get(name)
        if value is not None and isinstance(value, (int, float)):
            weighted_sum += float(value) * weight
            total_weight += weight
            included[name] = float(value)
    if total_weight > 0:
        composite = round(weighted_sum / total_weight, 4)
    else:
        composite = 0.0
    return {
        "composite_score": composite,
        "included_metrics": included,
        "total_weight_used": round(total_weight, 4),
        "tier": tier,
    }
