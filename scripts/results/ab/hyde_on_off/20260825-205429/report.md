# A/B 实验报告：HyDE 检索增强

- 实验：`hyde_on_off`
- 臂 A：**hyde_off**（关闭 HyDE）
- 臂 B：**hyde_on**（开启 HyDE）
- 测试集：`evaluation_qa_generated.md`；样本数：5；模式：full（6 项）；seed：7
- 臂执行顺序：hyde_off → hyde_on

| 指标 | 臂 A | 臂 B | Δ (B-A) | 相对变化 | 胜出 |
|---|---:|---:|---:|---:|---|
| 答案正确性 answer_correctness | N/A | N/A | N/A | N/A | — |
| 回答相关性 answer_relevancy | 0.9313 | 0.9232 | -0.0081 | -0.87% | A |
| 综合分 composite | 0.9088 | 0.9357 | 0.0269 | 2.96% | B |
| 实体召回 context_entity_recall | 0.1333 | 0.1924 | 0.0591 | 44.34% | B |
| 上下文精确率 context_precision | 0.6650 | 0.6650 | 0.0000 | 0.00% | tie |
| 上下文召回率 context_recall | 1.0000 | 1.0000 | 0.0000 | 0.00% | tie |
| 忠实度 faithfulness | 0.9000 | 1.0000 | 0.1000 | 11.11% | B |
| 命中率 hit_rate@5 | 1.0000 | 1.0000 | 0.0000 | 0.00% | tie |
| MRR@5 | 0.9000 | 0.9000 | 0.0000 | 0.00% | tie |
| multi_hop_hit | 0.0000 | 0.0000 | 0.0000 | N/A | tie |
| NDCG@5 | 1.9072 | 1.9072 | 0.0000 | 0.00% | tie |
