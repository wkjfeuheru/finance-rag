from finance_rag.src.rag.ingestion.chunker import HierarchicalChunker


def test_parents_follow_complete_heading_sections():
    chunker = HierarchicalChunker(max_tokens=512, parent_max_tokens=1)

    parents = chunker._split_into_parents(
        "# 第一章\n第一章正文。\n\n## 第一节\n第一节正文。\n\n# 第二章\n第二章正文。",
        "doc.md",
        "doc",
    )

    assert [parent.heading for parent in parents] == ["第一章", "第一节", "第二章"]
    assert parents[0].content == "# 第一章\n第一章正文。"
    assert parents[1].content == "## 第一节\n第一节正文。"
    assert parents[2].content == "# 第二章\n第二章正文。"


def test_unheaded_text_uses_natural_paragraphs_without_length_split():
    chunker = HierarchicalChunker(max_tokens=512, parent_max_tokens=1)
    text = "第一段第一行。\n第一段第二行。\n\n第二段内容。"

    parents = chunker._split_into_parents(text, "doc.txt", "doc")

    assert [parent.content for parent in parents] == [
        "第一段第一行。\n第一段第二行。",
        "第二段内容。",
    ]


def test_long_complete_paragraph_remains_one_parent_but_children_split():
    chunker = HierarchicalChunker(max_tokens=10, parent_max_tokens=1, overlap=0)
    paragraph = "。".join(["这是一个很长的自然段"] * 20) + "。"

    parents = chunker._split_into_parents(paragraph, "doc.txt", "doc")
    children = chunker._split_parent_into_children(parents[0].content)

    assert len(parents) == 1
    assert parents[0].content == paragraph
    assert len(children) > 1
    assert "这是一个很长的自然段" in "".join(children)


def test_tables_remain_atomic_in_child_processing():
    chunker = HierarchicalChunker(max_tokens=4, parent_max_tokens=1, overlap=0)
    table = "| 项目 | 数值 |\n| --- | --- |\n| 收入 | 100 |"

    children = chunker._split_parent_into_children(
        f"表格前文。\n\n{table}\n\n表格后文。"
    )

    assert table in children
