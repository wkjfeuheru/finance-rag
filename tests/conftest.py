"""测试全局约定。

**单测不得联网。** 仓库根的 ``.env`` 会带着真实模型 key 被加载，因此入库链路里
新增的任何模型调用都会让单测变成真实网络请求：耗时不可控、结果不可复现，
还会把时间敏感的流水线用例打挂。

这里默认关掉研报元数据的 LLM 抽取（只走正则）。需要验证 LLM 路径的用例，
显式给 ``enrich_chunks`` / ``extract_metadata`` 传 ``enable_llm=True`` 并
替换 ``_llm_extract``。

注意：``config`` 必须**在 fixture 内部**导入。若在模块级导入，config 会在
pytest 收集阶段就读掉环境变量，从而早于 ``test_auth`` 这类用例设置
``os.environ``（它们依赖「先设环境变量再导入」的顺序）。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_metadata_llm(monkeypatch):
    from finance_rag.src.core import config

    monkeypatch.setattr(config, "ENABLE_METADATA_LLM", False)
