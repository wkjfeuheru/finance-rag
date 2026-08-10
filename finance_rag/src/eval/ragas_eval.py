"""策略评估模块（基于 ragas 框架）。

使用 ragas 对不同检索策略（RRF 混合检索 / 纯向量检索、是否启用重排序）进行评估，
输出 faithfulness、answer_relevancy、context_precision、context_recall 等指标。

核心组件：
* :class:`StrategyConfig` — 策略配置（检索模式、重排序开关、检索深度）
* :class:`TestSetEntry` — 测试集条目（见 ``test_set.py``）
* :class:`StrategyEvaluator` — ragas 策略评估器
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
_LIBS_DIR = Path(__file__).resolve().parents[2] / "libs"
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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ragas.llms.base import LangchainLLMWrapper
from ragas.prompt.utils import extract_json

from config.settings import (
    CHAT_RERANK_TOP_K,
    CHAT_TOP_K,
    RAGAS_MAX_RETRIES,
    RAGAS_MAX_WORKERS,
    RAGAS_TIMEOUT_SECONDS,
    RRF_K,
    model,
)
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base
from finance_rag.src.eval.test_set import TestSetEntry, TestSetLoader, get_test_set_loader

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


def _compute_rejection_score(response: str, expected_behavior: str = "") -> float:
    """计算对抗层问题的拒绝分数。

    检测回答是否包含"未检索到"、"无法回答"、"暂无法"等拒绝表述。
    对于对抗性提问（adversarial），检查是否纠正了错误前提。

    Returns:
        1.0 表示正确拒绝/纠正，0.0 表示未正确响应。
    """
    if not response:
        return 0.0

    rejection_keywords = [
        "未检索到", "暂无法", "无法回答", "无法确定",
        "知识库中未", "暂无相关", "暂未检索",
    ]
    correction_keywords = [
        "不对", "不是", "并非", "不可以", "不能",
        "不存在", "不正确",
    ]

    has_rejection = any(kw in response for kw in rejection_keywords)
    has_correction = any(kw in response for kw in correction_keywords)

    if has_rejection:
        return 1.0
    if has_correction and len(response) > 20:
        return 0.8  # 有纠正意图但可能不够明确
    return 0.0


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
        use_dense_only: True 时只走稠密向量检索，False 时走 RRF 混合检索。
        use_rerank: 是否启用 BGE 重排序。
        rerank_top_n: 重排序返回的结果数。
        k: 检索深度（返回的上下文数）。
        rrf_k: RRF 融合常数。
    """

    use_dense_only: bool = False
    use_rerank: bool = False
    rerank_top_n: int = CHAT_RERANK_TOP_K
    k: int = CHAT_TOP_K
    rrf_k: int = RRF_K

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_dense_only": self.use_dense_only,
            "use_rerank": self.use_rerank,
            "rerank_top_n": self.rerank_top_n,
            "k": self.k,
            "rrf_k": self.rrf_k,
        }

    @property
    def label(self) -> str:
        """策略简短标签，用于结果展示。"""
        mode = "dense_only" if self.use_dense_only else "rrf_hybrid"
        return f"{mode}|rerank={'on' if self.use_rerank else 'off'}"


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
                "缺少标准答案，请检查 data/evaluation_qa.md 文件"
            )

        query = entry.query
        logger.info("评估 [%s]：%s", config.label, query[:50])

        # 1. 检索
        try:
            docs = self._kb.hybrid_search(
                query,
                k=config.k,
                use_dense_only=config.use_dense_only,
                expand_parents=True,
                use_rerank=config.use_rerank,
                rerank_top_n=config.rerank_top_n,
                rrf_k=config.rrf_k,
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
            # 计算本地检索指标
            retrieved_titles = [s.get("title", "") for s in (sources or [])]
            if entry.related_docs:
                local_metrics = _compute_local_retrieval_metrics(
                    retrieved_titles=retrieved_titles,
                    related_docs=entry.related_docs,
                    k=config.k,
                )
                metrics_dict.update(local_metrics)

            # 计算负面拒绝率（仅 L3 对抗层）
            if entry.test_type in ("negative", "adversarial", "ambiguous"):
                rejection_score = _compute_rejection_score(response, entry.expected_behavior)
                metrics_dict["negative_rejection"] = rejection_score

            # 计算综合评分
            composite = compute_composite_score(metrics_dict, tier=entry.test_type)
            metrics_dict["composite_score"] = composite["composite_score"]

            result_dict.update({
                "query": entry.query,
                "ground_truth": entry.ground_truth,
                "response": response,
                "sources": sources or [],
                "related_docs": entry.related_docs,
                "test_type": entry.test_type,
                "difficulty": entry.difficulty,
            })

        return result_dict

    @staticmethod
    def _get_metrics(llm=None, embeddings=None):
        """获取 ragas 评估指标列表。

        ragas 0.4 使用类形式指标，兼容实例形式。
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
            AnswerCorrectness(
                llm=llm,
                embeddings=embeddings,
            ),
            ContextEntityRecall(llm=llm),
        ]

        return metrics


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
