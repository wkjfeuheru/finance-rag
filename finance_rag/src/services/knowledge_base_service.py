"""知识库类别注册表：管理逻辑"知识库"（文档分类视图）的元数据。

设计约定（单集合多类别）：
* 物理上所有文档仍存放在同一个 Milvus 集合（``KB_COLLECTION_NAME``）中；
* "知识库"是逻辑概念 —— 每个知识库对应集合内 ``category`` 标量字段的一个取值；
* 内置四类（研报 / 政策 / 产品说明 / 交易或业务规则）首次运行时预置，
  此后与自定义类别同等对待：均可改名（联动迁移集合内文档分类）与删除；
* 用户可新增自定义类别，类别值直接写入上传文档的分类元数据；
* ``display_name`` / ``description`` 持久化在 PostgreSQL 注册表中；传入
  ``path`` 时仅为测试兼容而使用本地 JSON 文件。
"""

from __future__ import annotations

import logging
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finance_rag.src.core.config import ENABLE_VERSIONING, KB_COLLECTION_NAME
from finance_rag.src.infrastructure.relational_db.knowledge_base import KnowledgeBaseRepository
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_milvus_client
from finance_rag.src.rag.models.document_category import DOCUMENT_CATEGORIES

logger = logging.getLogger(__name__)

# 类别名规则：中文/字母/数字/下划线/连字符，长度 1-32（引号会破坏 Milvus 过滤表达式，单独校验）
_NAME_RE = re.compile(r"^\w[\w-]{0,31}$")


class KnowledgeBaseNotEmptyError(ValueError):
    """类别下仍有文档，禁止删除。"""


def _validate_name(name: str) -> str:
    """校验类别值并返回 strip 后的值；非法时抛 ValueError。"""
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            "类别名需以中文/字母/数字/下划线开头，仅含中文/字母/数字/下划线/连字符，长度 1-32"
        )
    if '"' in name or "'" in name:
        raise ValueError("类别名不能包含引号")
    return name


class KnowledgeBaseRegistry:
    """知识库类别注册表。

    默认使用 PostgreSQL 仓储；传入非空 ``path`` 时使用本地 JSON，供测试注入兼容。
    """

    def __init__(self, path: Path | None = None):
        self._path = Path(path) if path else None
        self._repository = None if path else KnowledgeBaseRepository()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 内部读写
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self._repository is not None:
            try:
                return self._repository.load()
            except Exception as exc:  # DB 不可用时降级为空列表，避免接口 500；seeded 置 True 防止触发播种写库
                logger.warning("读取知识库注册表（PostgreSQL）失败：%s", exc)
                return {"knowledge_bases": [], "builtin_seeded": True}
        assert self._path is not None
        if not self._path.exists():
            return {"knowledge_bases": []}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"knowledge_bases": []}
        if not isinstance(data.get("knowledge_bases"), list):
            return {"knowledge_bases": []}
        return data

    def _save(self, data: dict[str, Any]) -> None:
        if self._repository is not None:
            self._repository.save(data)
            return
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 原子替换，避免并发写坏文件
        tmp.replace(self._path)

    def _get_client(self) -> Any:
        """通过基础设施适配器获取 Milvus 客户端。"""
        return get_milvus_client()

    def _ensure_builtin_registered(self, data: dict[str, Any]) -> dict[str, Any]:
        """首次初始化时预置内置四类；此后不再自动补回已删除的类别。

        * 旧格式（物理集合条目 ``name == KB_COLLECTION_NAME``）会被迁移为内置四类，
          其余条目（含旧版自定义物理库）按自定义类别保留；
        * 预置状态由文件内 ``builtin_seeded`` 标记记录：一旦预置完成，
          用户删除的内置类别不会在下次加载时复活。
        """
        entries = list(data["knowledge_bases"])
        legacy = any(e.get("name") == KB_COLLECTION_NAME for e in entries)
        seeded = bool(data.get("builtin_seeded"))

        # 旧格式迁移：移除物理集合条目（finance_kb）
        if legacy:
            entries = [e for e in entries if e.get("name") != KB_COLLECTION_NAME]

        # 仅首次（未预置 / 旧格式迁移）补全内置四类
        if legacy or not seeded:
            existing = {e.get("name") for e in entries}
            for value, label in DOCUMENT_CATEGORIES.items():
                if value not in existing:
                    entries.append({
                        "name": value,
                        "display_name": label,
                        "description": "",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    })
            data["knowledge_bases"] = entries
            data["builtin_seeded"] = True
            self._save(data)
        return data

    # ------------------------------------------------------------------
    # 统计（单集合内按 category 过滤）
    # ------------------------------------------------------------------

    @staticmethod
    def _category_filter_expr(category: str) -> str:
        """构建 Milvus 过滤表达式：category 为集合顶层标量字段。

        版本保留模式（ENABLE_VERSIONING）下追加 ``is_current == true``，
        历史版本不计入类别统计。
        """
        expr = f'category == "{category}"'
        if ENABLE_VERSIONING:
            expr += " and is_current == true"
        return expr

    def _get_category_stats(self, name: str) -> dict[str, Any]:
        """统计某类别在默认集合中的文档数/切块数。

        返回结构含 ``ok`` 标志：``False`` 表示 Milvus 不可达或查询失败，
        此时计数不可信（0 不代表类别为空），删除等破坏性操作必须先检查该标志。
        """
        try:
            client = self._get_client()
            if not client.has_collection(KB_COLLECTION_NAME):
                return {"ok": True, "exists": False, "document_count": 0, "chunk_count": 0}

            expr = self._category_filter_expr(name)

            # 切块数：该类别下的向量行数（count(*) 不支持时回退迭代累计）
            chunk_count: int | None = None
            try:
                res = client.query(
                    KB_COLLECTION_NAME,
                    filter=expr,
                    output_fields=["count(*)"],
                )
                chunk_count = int(res[0].get("count(*)", 0)) if res else 0
            except Exception as exc:  # count(*) 不可用时回退逐行统计
                logger.debug("count(*) 统计失败，回退迭代统计：%s", exc)
                chunk_count = None

            # 文档数：source 去重；逐批迭代避免一次拉取超限
            sources: set[str] = set()
            iterator = client.query_iterator(
                collection_name=KB_COLLECTION_NAME,
                filter=expr,
                output_fields=["source"],
                batch_size=1000,
            )
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    for row in batch:
                        sources.add(row.get("source", ""))
                    if chunk_count is None:
                        chunk_count = (chunk_count or 0) + len(batch)
            finally:
                iterator.close()

            return {
                "ok": True,
                "exists": True,
                "document_count": len(sources),
                "chunk_count": int(chunk_count or 0),
            }
        except Exception as exc:  # Milvus 不可用等，仍返回注册表元数据（计数不可信）
            logger.warning("获取类别 %s 统计失败：%s", name, exc)
            return {"ok": False, "exists": False, "document_count": 0, "chunk_count": 0}

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        """列出所有知识库类别（内置四类 + 自定义），附文档数/切块数。"""
        with self._lock:
            data = self._ensure_builtin_registered(self._load())
            entries = list(data["knowledge_bases"])

        result: list[dict[str, Any]] = []
        for entry in entries:
            name = entry.get("name", "")
            stats = self._get_category_stats(name)
            result.append({
                "name": name,
                "display_name": entry.get("display_name", name),
                "description": entry.get("description", ""),
                "created_at": entry.get("created_at", ""),
                "document_count": int(stats.get("document_count", 0)),
                "chunk_count": int(stats.get("chunk_count", 0)),
            })
        return result

    # ------------------------------------------------------------------
    # 增删改
    # ------------------------------------------------------------------

    def create_knowledge_base(
        self,
        name: str,
        display_name: str = "",
        description: str = "",
    ) -> dict[str, Any]:
        """创建类别：校验名称 -> 查重 -> 写注册表。

        不创建 Milvus 集合：所有类别共用默认集合，仅作为分类元数据值存在。
        """
        name = _validate_name(name)

        with self._lock:
            data = self._ensure_builtin_registered(self._load())
            if any(kb.get("name") == name for kb in data["knowledge_bases"]):
                raise ValueError(f"类别 {name} 已存在")

            entry = {
                "name": name,
                "display_name": display_name.strip() or name,
                "description": description.strip(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            data["knowledge_bases"].append(entry)
            self._save(data)

        return {
            **entry,
            "document_count": 0,
            "chunk_count": 0,
        }

    def update_knowledge_base(
        self,
        name: str,
        display_name: str | None = None,
        description: str | None = None,
        new_name: str | None = None,
    ) -> dict[str, Any]:
        """修改类别的显示名/描述/类别值。

        ``new_name`` 非空且与 ``name`` 不同时执行改名：先把集合中
        ``category == name`` 的文档迁移到新类别值，再更新注册表条目。
        内置与自定义类别一视同仁。
        """
        rename_to = None
        if new_name is not None:
            rename_to = _validate_name(new_name)
            if rename_to == name:
                rename_to = None

        with self._lock:
            data = self._ensure_builtin_registered(self._load())
            target = next(
                (kb for kb in data["knowledge_bases"] if kb.get("name") == name),
                None,
            )
            if target is None:
                raise KeyError(f"类别 {name} 不存在")
            if rename_to and any(
                kb.get("name") == rename_to for kb in data["knowledge_bases"]
            ):
                raise ValueError(f"类别 {rename_to} 已存在")

            # 先迁移 Milvus 数据（失败则不写注册表，避免注册表与数据不一致）
            if rename_to:
                try:
                    self._migrate_category_value(name, rename_to)
                except Exception as exc:
                    logger.exception("迁移类别 %s -> %s 失败：%s", name, rename_to, exc)
                    raise RuntimeError(f"迁移类别数据失败：{exc}") from exc

            if rename_to:
                target["name"] = rename_to
            if display_name is not None:
                target["display_name"] = display_name.strip() or target["name"]
            if description is not None:
                target["description"] = description.strip()
            self._save(data)
            return dict(target)

    def _migrate_category_value(self, old: str, new: str) -> int:
        """将集合中 ``category == old`` 的行改为 ``new``（query + upsert）。

        返回迁移的行数；集合不存在或无误分类行时返回 0。
        """
        client = self._get_client()
        if not client.has_collection(KB_COLLECTION_NAME):
            return 0

        expr = self._category_filter_expr(old)
        rows: list[dict[str, Any]] = []
        iterator = client.query_iterator(
            collection_name=KB_COLLECTION_NAME,
            filter=expr,
            output_fields=["*"],
            batch_size=1000,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    row["category"] = new
                    rows.append(row)
        finally:
            iterator.close()

        if rows:
            client.upsert(KB_COLLECTION_NAME, rows)
            logger.info("类别迁移：%s -> %s，共 %d 行", old, new, len(rows))
        return len(rows)

    def delete_knowledge_base(self, name: str) -> dict[str, Any]:
        """删除类别：有文档拒绝 -> 仅移除注册表条目。

        内置与自定义类别均可删除；不 drop Milvus 集合（集合为所有类别共用）。
        Milvus 不可达时中止删除（无法确认类别是否为空）。
        """
        with self._lock:
            data = self._ensure_builtin_registered(self._load())
            entries = [kb for kb in data["knowledge_bases"] if kb.get("name") == name]
            if not entries:
                raise KeyError(f"类别 {name} 不存在")

            stats = self._get_category_stats(name)
            if not stats.get("ok"):
                raise RuntimeError(
                    "无法连接 Milvus，无法确认类别是否为空，删除已中止"
                )
            if stats.get("chunk_count", 0) > 0:
                raise KnowledgeBaseNotEmptyError(
                    f"类别 {entries[0].get('display_name', name)} 下还有 "
                    f"{stats.get('document_count', 0)} 个文档，请先删除该类别下的所有文档"
                )

            data["knowledge_bases"] = [
                kb for kb in data["knowledge_bases"] if kb.get("name") != name
            ]
            self._save(data)

        return {"name": name, "deleted": True}


# 全局单例
_registry: KnowledgeBaseRegistry | None = None


def get_kb_registry() -> KnowledgeBaseRegistry:
    """获取知识库类别注册表单例。"""
    global _registry
    if _registry is None:
        _registry = KnowledgeBaseRegistry()
    return _registry
