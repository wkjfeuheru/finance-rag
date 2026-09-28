"""存储后端选择。

历史 bug：``get_storage()`` 恒返回 OSS 后端，``STORAGE_BACKEND`` 只在生产配置校验里
被读到。后果不是「报错」而是**静默外发**——本地开发环境配了 ``local`` 仍会把
文档原件传到真实的公网对象存储。研报是授权材料，这个行为必须有测试钉住。
"""

from finance_rag.src.infrastructure import storage


def test_local_backend_is_selected_by_config(monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_BACKEND", "local")
    storage.reset_storage()

    backend = storage.get_storage()

    assert type(backend).__name__ == "LocalStorageBackend"


def test_object_storage_is_used_for_oss_and_s3(monkeypatch):
    for value in ("oss", "s3"):
        monkeypatch.setattr(storage, "STORAGE_BACKEND", value)
        storage.reset_storage()

        backend = storage.get_storage()

        assert type(backend).__name__ == "OSSStorageBackend", value


def test_backend_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_BACKEND", "  LOCAL ")
    storage.reset_storage()

    assert type(storage.get_storage()).__name__ == "LocalStorageBackend"


def test_backend_is_a_singleton_and_resettable(monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_BACKEND", "local")
    storage.reset_storage()

    first = storage.get_storage()
    assert storage.get_storage() is first

    storage.reset_storage()
    assert storage.get_storage() is not first


def test_local_backend_round_trips_through_upload_dir(monkeypatch, tmp_path):
    import asyncio

    from finance_rag.src.core import config
    from finance_rag.src.infrastructure.storage.local import LocalStorageBackend

    backend = LocalStorageBackend(root_dir=str(tmp_path))
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path))

    async def _exercise():
        await backend.upload("docs/x.md", "内容".encode("utf-8"))
        return await backend.download_text("docs/x.md")

    assert asyncio.run(_exercise()) == "内容"
    assert (tmp_path / "docs" / "x.md").read_text(encoding="utf-8") == "内容"
