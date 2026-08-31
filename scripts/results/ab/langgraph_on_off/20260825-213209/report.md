# A/B 实验报告：LangGraph Agentic RAG

- 实验：`langgraph_on_off`
- 臂 A：**standard**（标准 RAG）
- 臂 B：**agentic**（Agentic RAG）
- 测试集：`evaluation_qa_generated.md`；样本数：5；模式：full（6 项）；seed：7
- 臂执行顺序：standard → agentic

| 指标 | 臂 A | 臂 B | Δ (B-A) | 相对变化 | 胜出 |
|---|---:|---:|---:|---:|---|
| 答案正确性 answer_correctness | N/A | N/A | N/A | N/A | — |
| 回答相关性 answer_relevancy | 0.9353 | 0.9644 | 0.0291 | 3.11% | B |
| 综合分 composite | 0.9383 | 0.9293 | -0.0090 | -0.96% | A |
| 实体召回 context_entity_recall | 0.3333 | 0.0833 | -0.2500 | -75.01% | A |
| 上下文精确率 context_precision | 0.6650 | 0.7589 | 0.0939 | 14.12% | B |
| 上下文召回率 context_recall | 1.0000 | 1.0000 | 0.0000 | 0.00% | tie |
| 忠实度 faithfulness | 1.0000 | 0.9000 | -0.1000 | -10.00% | A |
| 命中率 hit_rate@5 | 1.0000 | 1.0000 | 0.0000 | 0.00% | tie |
| MRR@5 | 0.9000 | 1.0000 | 0.1000 | 11.11% | B |
| multi_hop_hit | 0.0000 | 0.0000 | 0.0000 | N/A | tie |
| NDCG@5 | 1.9072 | 2.0183 | 0.1111 | 5.83% | B |
