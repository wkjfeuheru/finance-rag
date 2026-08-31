# A/B 实验报告：动态 K

- 实验：`dynamic_k_on_off`
- 臂 A：**dynamic_k_off**（固定 K）
- 臂 B：**dynamic_k_on**（动态 K）
- 测试集：`evaluation_qa_generated.md`；样本数：3；模式：fast（3 项）；seed：42
- 臂执行顺序：dynamic_k_on → dynamic_k_off

| 指标 | 臂 A | 臂 B | Δ (B-A) | 相对变化 | 胜出 |
|---|---:|---:|---:|---:|---|
| 回答相关性 answer_relevancy | 0.5645 | 0.3088 | -0.2557 | -45.30% | A |
| 综合分 composite | 0.6792 | 0.4974 | -0.1818 | -26.77% | A |
| 上下文精确率 context_precision | 0.3333 | 0.3519 | 0.0186 | 5.58% | B |
| 忠实度 faithfulness | 0.7778 | 0.4603 | -0.3175 | -40.82% | A |
| 命中率 hit_rate@5 | 1.0000 | 1.0000 | 0.0000 | 0.00% | tie |
| MRR@5 | 1.0000 | 1.0000 | 0.0000 | 0.00% | tie |
| NDCG@5 | 2.6152 | 2.6529 | 0.0377 | 1.44% | B |
