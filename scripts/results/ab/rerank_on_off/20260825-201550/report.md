# A/B 实验报告：BGE 重排序

- 实验：`rerank_on_off`
- 臂 A：**rerank_off**（关闭重排序）
- 臂 B：**rerank_on**（开启重排序）
- 测试集：`evaluation_qa_generated.md`；样本数：5；模式：full（6 项）；seed：7
- 臂执行顺序：rerank_off → rerank_on

| 指标 | 臂 A | 臂 B | Δ (B-A) | 相对变化 | 胜出 |
|---|---:|---:|---:|---:|---|
| 答案正确性 answer_correctness | N/A | N/A | N/A | N/A | — |
| 回答相关性 answer_relevancy | 0.9337 | 0.9326 | -0.0011 | -0.12% | A |
| 综合分 composite | 0.9331 | 0.9377 | 0.0046 | 0.49% | B |
| 实体召回 context_entity_recall | 0.3333 | 0.3333 | 0.0000 | 0.00% | tie |
| 上下文精确率 context_precision | 0.6311 | 0.6650 | 0.0339 | 5.37% | B |
| 上下文召回率 context_recall | 1.0000 | 1.0000 | 0.0000 | 0.00% | tie |
| 忠实度 faithfulness | 1.0000 | 1.0000 | 0.0000 | 0.00% | tie |
| 命中率 hit_rate@5 | 1.0000 | 1.0000 | 0.0000 | 0.00% | tie |
| MRR@5 | 0.9000 | 0.9000 | 0.0000 | 0.00% | tie |
| multi_hop_hit | 0.0000 | 0.0000 | 0.0000 | N/A | tie |
| NDCG@5 | 1.8210 | 1.9072 | 0.0862 | 4.73% | B |
