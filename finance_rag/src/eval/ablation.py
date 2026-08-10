"""Paired ablation benchmark for the finance RAG pipeline.

The benchmark compares query rewriting, reranking, chunking and retrieval mode.
It deliberately lives outside the online chat path and uses isolated Milvus
collections for both chunking strategies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document

from .benchmark import (
    QUALITY_METRICS,
    build_sources,
    change,
    parse_relevant_sources,
    retrieval_metrics,
    safe_mean,
)
from config.settings import RRF_K
from .ragas_eval import TestSetEntry
from finance_rag.src.infrastructure.vector_store.milvus_kb import KnowledgeBase, ParsedDocument

RETRIEVAL_METRICS = ("hit_rate_at_5", "precision_at_5", "recall_at_5", "mrr_at_5", "evidence_recall_at_5")
DEFAULT_SEED = 20260725
DEFAULT_ROUNDS = 5
DEFAULT_SAMPLES_PER_ROUND = 5
DEFAULT_DOCLING_COLLECTION = "finance_benchmark_hierarchical_v1"
DEFAULT_RECURSIVE_COLLECTION = "finance_benchmark_recursive_v1"
LOCAL_PRECISION_QUESTION_IDS = (2, 4, 6, 8, 9, 10, 11, 14, 15, 16, 18, 21)
LOCAL_PRECISION_REQUIRED_FACTS = {
    2: ("每个自然年度最高缴费12000元", "未使用额度不能结转"),
    4: ("达到法定退休年龄可以领取", "可以一次性或分期领取"),
    6: ("IOPV每15秒更新一次", "溢价率等于市价除以IOPV减1"),
    8: ("申购使用一篮子证券", "最小申购赎回单位通常为50万份或100万份"),
    9: ("账户开通满2年以上", "前20个交易日日均资产不低于10万元"),
    10: ("T+2日公布中签结果", "T+2日16点前足额缴款"),
    11: ("赎回登记日前卖出或转股", "不处理会按约101元强制赎回"),
    14: ("估值低位加倍定投", "估值高位减少或暂停定投"),
    15: ("定投金额不超过月收入10%至20%", "不能影响正常生活"),
    16: ("达到预设收益率后全部赎回", "达到目标后分2至3批赎回"),
    18: ("普通股票涨跌幅限制为10%", "科创板和创业板涨跌幅限制为20%"),
    21: ("2023年GDP超过126万亿元", "同比增长5.2%"),
}


def select_dataset_entries(
    entries: Sequence[TestSetEntry], profile: str
) -> list[TestSetEntry]:
    if profile == "all":
        return list(entries)
    if profile == "hard":
        return [entry for entry in entries if entry.test_type == "multi_evidence"]
    if profile != "local_precision":
        raise ValueError(f"未知数据集配置：{profile}")

    by_id: dict[int, list[TestSetEntry]] = {}
    for entry in entries:
        if entry.question_id in LOCAL_PRECISION_REQUIRED_FACTS:
            by_id.setdefault(entry.question_id, []).append(entry)

    missing = [question_id for question_id in LOCAL_PRECISION_QUESTION_IDS if not by_id.get(question_id)]
    duplicates = [question_id for question_id in LOCAL_PRECISION_QUESTION_IDS if len(by_id.get(question_id, ())) > 1]
    if missing or duplicates:
        details = []
        if missing:
            details.append(f"缺少问题编号：{missing}")
        if duplicates:
            details.append(f"重复问题编号：{duplicates}")
        raise RuntimeError("；".join(details))

    selected_ids = set(LOCAL_PRECISION_QUESTION_IDS)
    return [
        replace(entry, required_facts=LOCAL_PRECISION_REQUIRED_FACTS[entry.question_id])
        for entry in entries
        if entry.question_id in selected_ids
    ]


@dataclass(frozen=True)
class AblationStrategy:
    name: str
    use_rewrite: bool = True
    use_rerank: bool = True
    chunking: str = "hierarchical"
    use_dense_only: bool = False
    k: int = 5
    rerank_top_n: int = 3
    rrf_k: int = RRF_K


@dataclass(frozen=True)
class Experiment:
    name: str
    label: str
    before: AblationStrategy
    after: AblationStrategy


OPTIMIZED = AblationStrategy(name="optimized")
EXPERIMENTS = (
    Experiment(
        "query_rewrite", "查询改写",
        replace(OPTIMIZED, name="关闭查询改写", use_rewrite=False),
        replace(OPTIMIZED, name="开启查询改写", use_rewrite=True),
    ),
    Experiment(
        "rerank", "重排序",
        replace(OPTIMIZED, name="关闭重排序", use_rerank=False),
        replace(OPTIMIZED, name="开启重排序", use_rerank=True),
    ),
    Experiment(
        "chunking", "切块策略",
        replace(OPTIMIZED, name="递归切块", chunking="recursive"),
        replace(OPTIMIZED, name="层级父子切块", chunking="hierarchical"),
    ),
    Experiment(
        "retrieval", "检索方式",
        replace(OPTIMIZED, name="纯向量检索", use_dense_only=True),
        replace(OPTIMIZED, name="RRF 混合检索", use_dense_only=False),
    ),
)


def select_experiments(names: Sequence[str] | None) -> tuple[Experiment, ...]:
    if names is None:
        return EXPERIMENTS
    if len(set(names)) != len(names):
        raise ValueError(f"实验名称重复：{list(names)}")
    by_name = {experiment.name: experiment for experiment in EXPERIMENTS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"未知实验：{unknown}")
    return tuple(by_name[name] for name in names)


def build_round_samples(
    entries: Sequence[Any], rounds: int = DEFAULT_ROUNDS,
    samples_per_round: int = DEFAULT_SAMPLES_PER_ROUND,
    seed: int = DEFAULT_SEED,
) -> list[list[Any]]:
    if rounds < 1:
        raise ValueError("rounds 必须大于 0")
    if samples_per_round < 1 or samples_per_round > len(entries):
        raise ValueError("samples_per_round 必须在 1 和测试集大小之间")
    rng = random.Random(seed)
    return [rng.sample(list(entries), samples_per_round) for _ in range(rounds)]


def evidence_recall(contexts: Sequence[str], required_facts: Sequence[str]) -> float | None:
    """Return the fraction of annotated facts supported by retrieved chunks."""
    if not required_facts:
        return None
    from .ragas_eval import _text_units
    context_units = [_text_units(context) for context in contexts]
    hits = 0
    for fact in required_facts:
        fact_units = _text_units(fact)
        coverage = max(
            (len(fact_units & units) / len(fact_units) for units in context_units),
            default=0.0,
        ) if fact_units else 0.0
        if coverage >= 0.6:
            hits += 1
    return round(hits / len(required_facts), 4)


class RecursiveDocumentChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_and_chunk(self, file_path, *, source, title):
        from finance_rag.src.infrastructure.vector_store.chunker import DoclingChunks, ParentChunk
        from finance_rag.src.infrastructure.parsing import get_parser
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        path = Path(file_path)
        parse_result = get_parser().parse(path, source=source, title=title)
        markdown = parse_result.markdown
        if not markdown or not markdown.strip():
            raise ValueError(f"文档文本为空：{source}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        texts = [text.strip() for text in splitter.split_text(markdown) if text.strip()]
        chunks = []
        import hashlib
        parent_id = hashlib.sha256(f"recursive:{source}".encode("utf-8")).hexdigest()
        for index, text in enumerate(texts):
            chunk_id = hashlib.sha256(f"recursive:{source}:{index}:{text}".encode("utf-8")).hexdigest()
            chunks.append(Document(
                page_content=text,
                metadata={"id": chunk_id, "source": source, "title": title, "chunk": index, "parent_id": parent_id},
            ))
        parents = [ParentChunk(id=parent_id, heading=source, content=markdown, source=source, title=title)]
        return DoclingChunks(chunks=chunks, markdown=markdown, parents=parents)


def make_knowledge_bases(
    data_dir: Path,
    hierarchical_collection: str = DEFAULT_DOCLING_COLLECTION,
    recursive_collection: str = DEFAULT_RECURSIVE_COLLECTION,
) -> dict[str, KnowledgeBase]:
    hierarchical = KnowledgeBase(collection_name=hierarchical_collection, docs_dir=str(data_dir))
    recursive = KnowledgeBase(collection_name=recursive_collection, docs_dir=str(data_dir))
    recursive._chunker = RecursiveDocumentChunker()
    return {"hierarchical": hierarchical, "recursive": recursive}


def source_files(data_dir: Path) -> list[Path]:
    extensions = {".pdf", ".md", ".txt", ".docx", ".pptx", ".html"}
    return sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def prepare_collections(kbs: dict[str, KnowledgeBase], data_dir: Path) -> dict[str, Any]:
    files = source_files(data_dir)
    if not files:
        raise ValueError(f"数据目录没有可入库文档：{data_dir}")
    result: dict[str, Any] = {}
    for chunking, kb in kbs.items():
        documents = []
        for path in files:
            documents.append(kb.add_document(path, source=path.name, title=path.stem))
        stats = kb.get_stats()
        expected = {path.name for path in files}
        actual = {item["source"] for item in kb.list_documents()}
        if expected - actual:
            raise RuntimeError(f"{chunking} 集合缺少来源：{sorted(expected - actual)}")
        if stats.get("chunk_count", 0) <= 0:
            raise RuntimeError(f"{chunking} 集合没有有效切块")
        result[chunking] = {
            "collection": kb.collection_name,
            "documents": documents,
            "stats": stats,
        }
    return result


class AblationBackend:
    def __init__(self, kbs: dict[str, KnowledgeBase]):
        from .ragas_eval import get_strategy_evaluator
        self.kbs = kbs
        self.evaluator = get_strategy_evaluator()
        self._rewrite_cache: dict[str, str] = {}

    def rewritten_query(self, query: str, enabled: bool) -> str:
        if not enabled:
            return query
        if query not in self._rewrite_cache:
            from finance_rag.src.application.chat_service import rewrite_query
            self._rewrite_cache[query] = rewrite_query(query)
        return self._rewrite_cache[query]

    def retrieve(self, strategy: AblationStrategy, query: str) -> list[dict[str, Any]]:
        kb = self.kbs[strategy.chunking]
        if strategy.use_dense_only:
            return self._dense_only_search(kb, query, strategy)
        return kb.hybrid_search(
            query, k=strategy.k,
            use_dense_only=False,
            expand_parents=True,
            use_rerank=strategy.use_rerank,
            rerank_top_n=strategy.rerank_top_n,
            rrf_k=strategy.rrf_k,
        )

    @staticmethod
    def _dense_only_search(
        kb: KnowledgeBase, query: str, strategy: AblationStrategy
    ) -> list[dict[str, Any]]:
        """Dense-only Milvus search; no sparse request is constructed."""
        from finance_rag.src.infrastructure.vector_store.hybrid_retriever import BGEReranker, _OVERSAMPLE_FACTOR

        kb.ensure_collection()
        fetch_k = (
            max(strategy.k, strategy.rerank_top_n * _OVERSAMPLE_FACTOR)
            if strategy.use_rerank else strategy.k
        )
        limit = fetch_k * _OVERSAMPLE_FACTOR
        results = kb._get_client().search(
            collection_name=kb.collection_name,
            data=[kb._embed_query(query)],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=["content", "source", "title", "chunk", "parent_id"],
        )
        matches = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            matches.append({
                "content": entity.get("content"),
                "source": entity.get("source"),
                "title": entity.get("title") or "金融文档",
                "chunk": entity.get("chunk"),
                "parent_id": entity.get("parent_id", ""),
                "score": float(hit.get("distance", hit.get("score", 0.0))),
            })
        if strategy.use_rerank and matches:
            return BGEReranker().rerank(
                query, matches, top_n=max(strategy.k, strategy.rerank_top_n)
            )
        return matches[:strategy.k]

    def generate(self, original_query: str, contexts: list[str]) -> str:
        from .ragas_eval import _ANSWER_PROMPT, model
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        if model is None:
            raise RuntimeError("LLM 未配置：请设置 DEEPSEEK_API_KEY")
        chain = ChatPromptTemplate.from_messages([("human", _ANSWER_PROMPT)]) | model | StrOutputParser()
        return chain.invoke({
            "context": self.evaluator._build_context(contexts),
            "query": original_query,
        }).strip()

    def evaluate_quality(
        self, strategy: AblationStrategy, entry: Any,
        contexts: list[str], sources: list[dict[str, Any]], response: str,
    ) -> dict[str, Any]:
        from .ragas_eval import StrategyConfig
        config = StrategyConfig(
            use_dense_only=strategy.use_dense_only,
            use_rerank=strategy.use_rerank,
            rerank_top_n=strategy.rerank_top_n,
            k=strategy.k,
        )
        data = [{
            "user_input": entry.query,
            "retrieved_contexts": contexts,
            "response": response,
            "reference": entry.ground_truth,
        }]
        return self.evaluator._run_ragas(data, config, entry, sources, response)


class AblationRunner:
    def __init__(
        self, backend: AblationBackend,
        experiments: Sequence[Experiment] = EXPERIMENTS,
    ):
        self.backend = backend
        self.experiments = tuple(experiments)

    def run(self, rounds: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for experiment in self.experiments:
            for round_index, entries in enumerate(rounds, 1):
                for sample_index, entry in enumerate(entries, 1):
                    # Alternate arm order to reduce warm-cache/order bias.
                    arms = (("before", experiment.before), ("after", experiment.after))
                    if (round_index + sample_index) % 2:
                        arms = tuple(reversed(arms))
                    for arm, strategy in arms:
                        rows.append(self._run_one(
                            experiment, arm, strategy, round_index, sample_index, entry
                        ))
        return rows

    def _run_one(
        self, experiment: Experiment, arm: str, strategy: AblationStrategy,
        round_index: int, sample_index: int, entry: Any,
    ) -> dict[str, Any]:
        started_all = time.perf_counter()
        rewritten = self.backend.rewritten_query(entry.query, strategy.use_rewrite)
        row: dict[str, Any] = {
            "experiment": experiment.name, "experiment_label": experiment.label,
            "arm": arm, "strategy": strategy.name,
            "strategy_config": asdict(strategy), "round": round_index,
            "sample": sample_index, "query": entry.query,
            "rewritten_query": rewritten, "related_docs": entry.related_docs,
            "retrieval": {}, "quality": {name: None for name in QUALITY_METRICS},
            "sources": [], "response": "", "errors": [],
        }
        started = time.perf_counter()
        try:
            docs = self.backend.retrieve(strategy, rewritten)
        except Exception as exc:
            docs = []
            row["errors"].append({"stage": "retrieval", "message": str(exc)})
        row["retrieval_latency_ms"] = round((time.perf_counter() - started) * 1000, 4)
        row["sources"] = build_sources(docs)
        row["retrieval"] = retrieval_metrics(
            [item["source"] for item in row["sources"]],
            parse_relevant_sources(entry.related_docs), strategy.k,
        )
        contexts = [doc.get("content", "") for doc in docs if doc.get("content")]
        row["retrieval"]["evidence_recall_at_5"] = evidence_recall(
            contexts[:strategy.k], getattr(entry, "required_facts", ())
        )
        started = time.perf_counter()
        try:
            row["response"] = self.backend.generate(entry.query, contexts)
        except Exception as exc:
            row["errors"].append({"stage": "generation", "message": str(exc)})
        row["generation_latency_ms"] = round((time.perf_counter() - started) * 1000, 4)
        started = time.perf_counter()
        if row["response"]:
            try:
                evaluated = self.backend.evaluate_quality(
                    strategy, entry, contexts, row["sources"], row["response"]
                )
                row["quality"].update(evaluated.get("metrics", {}))
                row["metric_errors"] = evaluated.get("metric_errors", {})
            except Exception as exc:
                row["errors"].append({"stage": "ragas", "message": str(exc)})
        row["ragas_latency_ms"] = round((time.perf_counter() - started) * 1000, 4)
        row["end_to_end_latency_ms"] = round((time.perf_counter() - started_all) * 1000, 4)
        row["success"] = not row["errors"]
        return row


def _round_metric(rows: Sequence[dict[str, Any]], section: str, metric: str) -> float | None:
    return safe_mean([row.get(section, {}).get(metric) for row in rows])


def summarize_ablation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    included = {row["experiment"] for row in rows}
    for experiment in (item for item in EXPERIMENTS if item.name in included):
        exp_rows = [row for row in rows if row["experiment"] == experiment.name]
        arms: dict[str, Any] = {}
        for arm in ("before", "after"):
            arm_rows = [row for row in exp_rows if row["arm"] == arm]
            round_summaries = []
            for round_index in sorted({row["round"] for row in arm_rows}):
                current = [row for row in arm_rows if row["round"] == round_index]
                retrieval = {m: _round_metric(current, "retrieval", m) for m in RETRIEVAL_METRICS}
                quality = {m: _round_metric(current, "quality", m) for m in QUALITY_METRICS}
                quality["quality_composite"] = safe_mean(list(quality.values()))
                round_summaries.append({
                    "round": round_index, "sample_count": len(current),
                    "retrieval": retrieval, "quality": quality,
                })
            retrieval = {
                m: safe_mean([item["retrieval"][m] for item in round_summaries])
                for m in RETRIEVAL_METRICS
            }
            quality_names = (*QUALITY_METRICS, "quality_composite")
            quality = {
                m: safe_mean([item["quality"][m] for item in round_summaries])
                for m in quality_names
            }
            arms[arm] = {
                "strategy": experiment.before.name if arm == "before" else experiment.after.name,
                "sample_count": len(arm_rows),
                "valid_pair_candidates": sum(row["success"] for row in arm_rows),
                "rounds": round_summaries, "retrieval": retrieval, "quality": quality,
                "error_count": sum(bool(row["errors"]) for row in arm_rows),
            }
        comparison = {}
        paired_rows: dict[str, dict[tuple[int, int], dict[str, Any]]] = {
            "before": {}, "after": {},
        }
        for row_index, row in enumerate(exp_rows, 1):
            key = (row["round"], row.get("sample", row_index))
            paired_rows[row["arm"]][key] = row
        for section, metrics in (("retrieval", RETRIEVAL_METRICS),
                                 ("quality", (*QUALITY_METRICS, "quality_composite"))):
            comparison[section] = {}
            for metric in metrics:
                def metric_value(row: dict[str, Any]) -> float | None:
                    if metric == "quality_composite":
                        return safe_mean(list(row.get("quality", {}).values()))
                    return row.get(section, {}).get(metric)

                before_values = {
                    key: metric_value(row) for key, row in paired_rows["before"].items()
                }
                after_values = {
                    key: metric_value(row) for key, row in paired_rows["after"].items()
                }
                pair_keys = before_values.keys() & after_values.keys()
                valid_pairs = sum(
                    before_values[key] is not None and after_values[key] is not None
                    for key in pair_keys
                )
                before_total = len(before_values)
                after_total = len(after_values)
                comparison[section][metric] = {
                    "before": arms["before"][section][metric],
                    "after": arms["after"][section][metric],
                    **change(
                        arms["before"][section][metric],
                        arms["after"][section][metric],
                    ),
                    "valid_pairs": valid_pairs,
                    "before_missing_rate": (
                        sum(value is None for value in before_values.values()) / before_total
                        if before_total else None
                    ),
                    "after_missing_rate": (
                        sum(value is None for value in after_values.values()) / after_total
                        if after_total else None
                    ),
                }
        summary[experiment.name] = {
            "label": experiment.label, **arms, "comparison": comparison,
        }
    return summary


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def render_report(metadata: dict[str, Any], summary: dict[str, Any]) -> str:
    metric_labels = {
        "precision_at_5": "检索精确率@5", "recall_at_5": "检索召回率@5",
        "hit_rate_at_5": "命中率@5", "mrr_at_5": "MRR@5",
        "evidence_recall_at_5": "证据事实召回率@5",
        "context_precision": "上下文精确率", "context_recall": "上下文召回率",
        "faithfulness": "幻觉抑制/忠实度", "answer_relevancy": "回答相关性",
        "quality_composite": "质量综合分",
    }
    lines = [
        "# RAG 优化方案配对消融报告", "",
        f"- 运行时间：{metadata['created_at']}",
        f"- 轮数：{metadata['rounds']}；每轮样本：{metadata['samples_per_round']}",
        f"- 随机种子：{metadata['seed']}", "",
    ]
    for experiment in (item for item in EXPERIMENTS if item.name in summary):
        item = summary[experiment.name]
        lines.extend([
            f"## {experiment.label}：{experiment.before.name} → {experiment.after.name}", "",
            "| 指标 | 优化前 | 优化后 | 绝对变化 | 相对变化 | 有效配对数 | 优化前缺失率 | 优化后缺失率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for section in ("retrieval", "quality"):
            for metric, result in item["comparison"][section].items():
                relative = "N/A" if result["relative_pct"] is None else f"{result['relative_pct']:.2f}%"
                before_missing = "N/A" if result["before_missing_rate"] is None else f"{result['before_missing_rate']:.2%}"
                after_missing = "N/A" if result["after_missing_rate"] is None else f"{result['after_missing_rate']:.2%}"
                lines.append(
                    f"| {metric_labels.get(metric, metric)} | {_fmt(result['before'])} | "
                    f"{_fmt(result['after'])} | {_fmt(result['absolute'])} | {relative} | "
                    f"{result['valid_pairs']} | {before_missing} | {after_missing} |"
                )
        lines.extend([
            "", f"- 优化前错误样本：{item['before']['error_count']}；"
            f"优化后错误样本：{item['after']['error_count']}", "",
        ])
    return "\n".join(lines) + "\n"


def write_ablation_reports(
    output_dir: Path, metadata: dict[str, Any],
    rows: Sequence[dict[str, Any]], summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_results.json").write_text(
        json.dumps({"metadata": metadata, "results": list(rows)}, ensure_ascii=False,
                   indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "experiments": summary}, ensure_ascii=False,
                   indent=2, allow_nan=False), encoding="utf-8"
    )
    fields = [
        "experiment", "arm", "strategy", "round", "sample", "query",
        "rewritten_query", *RETRIEVAL_METRICS, *QUALITY_METRICS,
        "retrieval_latency_ms", "generation_latency_ms", "ragas_latency_ms",
        "end_to_end_latency_ms", "success", "errors",
    ]
    with (output_dir / "per_query.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{field: row.get(field) for field in fields},
                **row["retrieval"], **row["quality"],
                "errors": json.dumps(row["errors"], ensure_ascii=False),
            })
    (output_dir / "report.md").write_text(render_report(metadata, summary), encoding="utf-8")


def default_output_dir(root: Path) -> Path:
    return root / "artifacts" / "benchmarks" / datetime.now().strftime("%Y%m%d-%H%M%S-ablation")


def metadata_for_run(
    rounds: int, samples_per_round: int, seed: int,
    collections: dict[str, Any] | None = None,
    experiments: Sequence[Experiment] = EXPERIMENTS,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "rounds": rounds, "samples_per_round": samples_per_round, "seed": seed,
        "total_evaluations": len(experiments) * 2 * rounds * samples_per_round,
        "collections": collections or {},
        "experiments": [
            {"name": exp.name, "before": asdict(exp.before), "after": asdict(exp.after)}
            for exp in experiments
        ],
    }
