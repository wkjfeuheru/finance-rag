"""策略评估 API 路由（ragas）。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from config.settings import RAGAS_MAX_WORKERS
from finance_rag.src.utils.audit import log as audit_log
from finance_rag.src.api.dependencies import get_current_user
from finance_rag.src.eval.ragas_eval import (
    StrategyConfig,
    get_strategy_evaluator,
    get_test_set_loader,
)
from finance_rag.src.schemas.eval import (
    EvaluateStrategyRequest,
    EvaluateSingleQueryRequest,
    BatchEvaluateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_EVAL_RESULTS_DIR = Path(__file__).resolve().parents[3] / "eval" / "results"


def _filter_by_tier(entries: list, tier: str) -> list:
    """按测试层级过滤测试集条目。"""
    if tier == "L1":
        return [e for e in entries if e.test_type == "standard" and e.difficulty != "hard"]
    if tier == "L2":
        return [e for e in entries if e.test_type == "multi_evidence" or e.difficulty == "hard"]
    if tier == "L3":
        return [e for e in entries if e.test_type in ("negative", "adversarial", "ambiguous")]
    return entries  # all


def _save_eval_run(run_id: str, summary: dict, results: list) -> None:
    """保存评估运行结果到磁盘。"""
    run_dir = Path(_EVAL_RESULTS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 更新索引
    index_path = Path(_EVAL_RESULTS_DIR) / "index.json"
    index_entries = []
    if index_path.exists():
        try:
            index_entries = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index_entries = []
    index_entries.insert(0, {
        "run_id": run_id,
        "tier": summary.get("tier", ""),
        "total": summary.get("total", 0),
        "success_count": summary.get("success_count", 0),
        "elapsed_seconds": summary.get("elapsed_seconds", 0),
        "timestamp": summary.get("timestamp", ""),
    })
    # 只保留最近 50 条
    index_path.write_text(
        json.dumps(index_entries[:50], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@router.get("/test-queries")
async def get_test_queries(
    current_user: Annotated[str, Depends(get_current_user)],
):
    """获取测试查询列表及标准答案状态。

    标准答案已固化在 ``finance_rag/eval/data/evaluation_qa.md``，无需调用大模型生成。
    """
    try:
        from finance_rag.src.eval.test_set import get_test_set_loader
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="ragas 评估依赖未安装。请运行 pip install ragas==0.4.3 datasets>=4.0.0",
        )

    loader = get_test_set_loader()
    test_set = loader.load_test_set()
    has_gt = loader.has_ground_truth()

    return {
        "queries": [entry.to_dict() for entry in test_set],
        "count": len(test_set),
        "has_ground_truth": has_gt,
    }


@router.post("/evaluate-strategy")
async def evaluate_strategy(
    req: EvaluateSingleQueryRequest,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """评估前端手动输入的单条测试查询并返回 Ragas 指标。"""
    loader = get_test_set_loader()
    if not loader.has_ground_truth():
        raise HTTPException(
            status_code=400,
            detail="标准答案缺失，请检查 finance_rag/eval/data/evaluation_qa.md 文件",
        )

    query = req.query.strip()
    test_set = loader.load_test_set()
    entry = next((item for item in test_set if item.query.strip() == query), None)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail="未找到该查询对应的标准答案，请先将查询及标准答案添加到测试数据文件中",
        )

    config = StrategyConfig(
        use_dense_only=req.use_dense_only,
        use_rerank=req.use_rerank,
        rerank_top_n=req.rerank_top_n,
        k=req.k,
    )
    try:
        logger.info("评估手动输入的测试查询 query=%s", query)
        result = get_strategy_evaluator().evaluate(config, entry)
        audit_log("evaluate", user=current_user, resource=query[:100], detail=f"k={req.k}")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("策略评估失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/evaluate-batch")
async def evaluate_batch(
    req: BatchEvaluateRequest,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """批量评估：按层级选择测试集，并行执行评估并返回汇总报告。

    返回 eval_run_id，可通过 /api/eval-runs/{id} 查看详情。
    """
    loader = get_test_set_loader()
    test_set = loader.load_test_set()
    entries = _filter_by_tier(test_set, req.tier)

    if not entries:
        raise HTTPException(status_code=400, detail=f"层级 {req.tier} 没有可用的测试用例")

    config = StrategyConfig(
        use_dense_only=req.use_dense_only,
        use_rerank=req.use_rerank,
        rerank_top_n=req.rerank_top_n,
        k=req.k,
    )

    run_id = uuid.uuid4().hex[:12]
    evaluator = get_strategy_evaluator()
    semaphore = asyncio.Semaphore(RAGAS_MAX_WORKERS)

    async def evaluate_one(entry):
        async with semaphore:
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, evaluator.evaluate, config, entry)
                return {"success": True, "entry": entry.to_dict(), "result": result}
            except Exception as exc:
                logger.error("批量评估失败 [%s]: %s", entry.query[:30], exc)
                return {"success": False, "entry": entry.to_dict(), "error": str(exc)}

    start_time = time.perf_counter()
    results = await asyncio.gather(*[evaluate_one(e) for e in entries])
    elapsed = time.perf_counter() - start_time

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    # 汇总指标
    all_metrics: dict[str, list[float]] = {}
    for r in successes:
        for name, value in (r["result"].get("metrics", {}) or {}).items():
            if value is not None:
                all_metrics.setdefault(name, []).append(float(value))

    summary = {
        "run_id": run_id,
        "tier": req.tier,
        "strategy": config.to_dict(),
        "total": len(entries),
        "success_count": len(successes),
        "failure_count": len(failures),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregated_metrics": {
            name: round(sum(vals) / len(vals), 4) if vals else None
            for name, vals in all_metrics.items()
        },
    }

    # 持久化结果
    _save_eval_run(run_id, summary, results)

    audit_log(
        "evaluate_batch",
        user=current_user,
        resource=req.tier,
        detail=f"成功 {len(successes)}/{len(entries)}，耗时 {elapsed:.1f}s",
    )

    return {
        "run_id": run_id,
        "summary": summary,
        "results": [{
            "question_id": r["entry"]["question_id"],
            "query": r["entry"]["query"],
            "success": r["success"],
            "metrics": r.get("result", {}).get("metrics") if r["success"] else None,
            "error": r.get("error"),
        } for r in results],
    }


@router.get("/eval-runs")
async def list_eval_runs(
    current_user: Annotated[str, Depends(get_current_user)],
):
    """列出历史评估运行记录。"""
    index_path = Path(_EVAL_RESULTS_DIR) / "index.json"
    if not index_path.exists():
        return {"runs": [], "count": 0}

    try:
        runs = json.loads(index_path.read_text(encoding="utf-8"))
        return {"runs": runs, "count": len(runs)}
    except Exception:
        return {"runs": [], "count": 0}


@router.get("/eval-runs/{run_id}")
async def get_eval_run(
    run_id: str,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """获取单次评估运行的详细结果。"""
    run_dir = Path(_EVAL_RESULTS_DIR) / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="评估运行记录不存在")

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    return {"run_id": run_id, "summary": summary, "results": results}
