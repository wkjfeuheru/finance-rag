"""RAG 优化点 A/B 实验目录（配置开关驱动的实验定义）。

每个实验 = 一对臂（基线 vs 优化）。臂以环境变量集合（``env_overrides``）
声明，沿用项目配置开关形式；执行时由 runner 在独立子进程中应用，
与 ``.env``/默认值叠加（进程环境变量优先级最高）。

所有实验均为运行时实验，共享默认集合 ``finance_kb``，开关在检索/生成时生效。

注：
* 内容清洗（规则噪声过滤）与 SimHash 去重（文档内段落级 / 文档间）已固化
  为入库固定流程，非配置开关，不设对应实验；
* 语义分块、版本保留等入库时功能影响的是集合内容本身，且离线索引入口已移除，
  不设 A/B 实验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ABArm:
    """A/B 实验的单臂定义。

    ``kind`` 区分运行时实验（共享默认集合）与入库实验（每臂独立集合，
    实验前按臂开关重新索引 ``files/`` 文档）。
    """

    name: str
    label: str
    env_overrides: dict[str, str] = field(default_factory=dict)
    strategy_overrides: dict[str, Any] = field(default_factory=dict)
    kind: str = "runtime"  # runtime | ingest
    retrieval: str = "default"  # default | langgraph

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "retrieval": self.retrieval,
            "env_overrides": dict(self.env_overrides),
            "strategy_overrides": dict(self.strategy_overrides),
        }


@dataclass(frozen=True)
class ABExperiment:
    """A/B 实验：一对臂的对比定义。"""

    name: str
    label: str
    arm_a: ABArm  # 基线
    arm_b: ABArm  # 优化

    @property
    def kind(self) -> str:
        """任一臂为入库实验则整体视为入库实验（需独立集合索引）。"""
        return "ingest" if (
            self.arm_a.kind == "ingest" or self.arm_b.kind == "ingest"
        ) else "runtime"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "arm_a": self.arm_a.to_dict(),
            "arm_b": self.arm_b.to_dict(),
        }


def _arm(
    name: str,
    label: str,
    *,
    kind: str = "runtime",
    env: dict[str, str] | None = None,
    strategy: dict[str, Any] | None = None,
    retrieval: str = "default",
) -> ABArm:
    return ABArm(
        name=name,
        label=label,
        kind=kind,
        retrieval=retrieval,
        env_overrides=env or {},
        strategy_overrides=strategy or {},
    )


# ---------------------------------------------------------------------------
# 内置实验目录
# ---------------------------------------------------------------------------

AB_EXPERIMENTS: tuple[ABExperiment, ...] = (
    ABExperiment(
        "hybrid_vs_dense", "检索方式（稠密 vs RRF 混合）",
        _arm("dense", "纯稠密检索", strategy={"use_dense_only": True}),
        _arm("hybrid", "RRF 混合检索", strategy={"use_dense_only": False}),
    ),
    ABExperiment(
        "rerank_on_off", "BGE 重排序",
        _arm("rerank_off", "关闭重排序", env={"ENABLE_RERANKER": "false"}),
        _arm("rerank_on", "开启重排序", env={"ENABLE_RERANKER": "true"}),
    ),
    ABExperiment(
        "dynamic_k_on_off", "动态 K",
        _arm("dynamic_k_off", "固定 K", env={"DYNAMIC_K": "false"}),
        _arm("dynamic_k_on", "动态 K", env={"DYNAMIC_K": "true"}),
    ),
    ABExperiment(
        "hyde_on_off", "HyDE 检索增强",
        _arm("hyde_off", "关闭 HyDE", env={"ENABLE_HYDE": "false"}),
        _arm("hyde_on", "开启 HyDE", env={"ENABLE_HYDE": "true"}),
    ),
    ABExperiment(
        "langgraph_on_off", "LangGraph Agentic RAG",
        _arm("standard", "标准 RAG", env={"ENABLE_LANGGRAPH": "false"}),
        _arm("agentic", "Agentic RAG", env={"ENABLE_LANGGRAPH": "true"}, retrieval="langgraph"),
    ),
    ABExperiment(
        "query_rewrite_on_off", "查询改写 + 多路子查询",
        _arm("rewrite_off", "关闭改写", env={"CHAT_ENABLE_QUERY_REWRITE": "false"}),
        _arm("rewrite_on", "开启改写", env={"CHAT_ENABLE_QUERY_REWRITE": "true"}),
    ),
    ABExperiment(
        "semantic_chunk_on_off", "语义切块（语义+层级 vs 仅层级）",
        _arm("hierarchical", "仅层级切块", kind="ingest",
             env={"ENABLE_SEMANTIC_CHUNKER": "false"}),
        _arm("semantic", "语义+层级切块", kind="ingest",
             env={"ENABLE_SEMANTIC_CHUNKER": "true"}),
    ),
)


def select_experiments(names: list[str] | None) -> tuple[ABExperiment, ...]:
    """按名称选择实验；``names=None`` 时返回全部。"""
    if not names:
        return AB_EXPERIMENTS
    if len(set(names)) != len(names):
        raise ValueError(f"实验名称重复：{names}")
    by_name = {experiment.name: experiment for experiment in AB_EXPERIMENTS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"未知实验：{unknown}")
    return tuple(by_name[name] for name in names)
