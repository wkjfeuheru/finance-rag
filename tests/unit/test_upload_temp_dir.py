"""上传暂存文件的落点。

实测踩到的失败：上传把原件写到 ``tempfile.gettempdir()``，入队后文件被
OS/沙箱的临时目录清理扫掉，解析阶段报
``[Errno 2] No such file or directory: '...<hash>.uploading.pdf'``——
最终表现为「某些文档入库失败」，且失败点在解析而不是上传，很难联想到临时目录。

因此暂存文件必须落在**自己的存储目录**下：与真实产物同域，既可写也可排障。
"""

from pathlib import Path

from finance_rag.src.services import document_service


def test_upload_temp_dir_is_under_upload_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(document_service, "UPLOAD_DIR", str(tmp_path))

    resolved = document_service._upload_temp_dir()

    assert resolved == tmp_path / ".uploading"
    assert resolved.is_dir()


def test_upload_temp_dir_is_created_once_and_reusable(monkeypatch, tmp_path):
    monkeypatch.setattr(document_service, "UPLOAD_DIR", str(tmp_path))

    first = document_service._upload_temp_dir()
    second = document_service._upload_temp_dir()

    assert first == second


def test_production_default_is_not_the_system_temp_dir():
    """核心断言：按生产默认配置（UPLOAD_DIR=<repo>/files）不再往系统临时目录写。

    注意不能用 pytest 的 tmp_path 来验这条——tmp_path 本身就位于系统临时目录下，
    那样断言的是测试夹具而不是生产行为。
    """
    import tempfile

    resolved = document_service._upload_temp_dir().resolve()
    system_temp = Path(tempfile.gettempdir()).resolve()

    assert resolved != system_temp
    assert system_temp not in resolved.parents
    assert resolved.parent == Path(document_service.UPLOAD_DIR).resolve()
