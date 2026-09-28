# A/B 对比：hybrid_vs_dense

- 判定：inconclusive
- 主指标：recall_at_5
- 护栏：precision_at_5, mrr
- Bootstrap：10000 次（seed=7）

## Gold（n=18）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|
| mrr | 0.1111 | 0.1111 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| precision_at_5 | 0.0333 | 0.0333 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| recall_at_5 | 0.1389 | 0.1389 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |

## Silver（n=0）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|

