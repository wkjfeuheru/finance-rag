<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { getTestQueries, evaluateStrategy } from '../api'

// 测试集状态
const testQueries = ref([])
const hasGroundTruth = ref(false)
const loadingQueries = ref(false)

// 手动输入的查询文本
const queryText = ref('')

// 策略配置
const denseWeight = ref(0.7)
const useRerank = ref(false)
const rerankTopN = ref(3)
const retrievalK = ref(5)

const sparseWeight = computed(() => Math.round((1 - denseWeight.value) * 10) / 10)

// 评估结果
const evaluating = ref(false)
const evalResult = ref(null)
const resultHistory = ref([])

// 指标中文名称
const metricLabels = {
  faithfulness: '忠实度',
  answer_relevancy: '回答相关性',
  context_precision: '上下文精确度',
  context_recall: '上下文召回率'
}

// 输入内容与测试集完全匹配时展示对应元数据；标准答案仍以测试集为准。
const selectedEntry = computed(() => {
  const query = queryText.value.trim()
  return testQueries.value.find(item => item.query.trim() === query) || null
})

async function loadTestQueries() {
  loadingQueries.value = true
  try {
    const data = await getTestQueries()
    testQueries.value = data.queries || []
    hasGroundTruth.value = data.has_ground_truth || false
  } catch (e) {
    ElMessage.error('加载测试查询失败：' + (e.response?.data?.detail || e.message))
  } finally {
    loadingQueries.value = false
  }
}

async function handleEvaluate() {
  if (!hasGroundTruth.value) {
    ElMessage.warning('标准答案缺失，请检查 files/docs/evaluation_qa.md 文件')
    return
  }
  const query = queryText.value.trim()
  if (!query) {
    ElMessage.warning('请输入测试查询')
    return
  }
  if (!selectedEntry.value) {
    ElMessage.warning('未找到对应的标准答案，请先将该查询添加到 files/docs/evaluation_qa.md')
    return
  }

  evaluating.value = true
  evalResult.value = null

  try {
    const payload = {
      query,
      dense_weight: denseWeight.value,
      sparse_weight: sparseWeight.value,
      use_rerank: useRerank.value,
      rerank_top_n: rerankTopN.value,
      k: retrievalK.value
    }

    const result = await evaluateStrategy(payload)
    evalResult.value = result
    resultHistory.value.unshift(result)

    ElMessage.success('评估完成')
  } catch (e) {
    ElMessage.error('评估失败：' + (e.response?.data?.detail || e.message))
  } finally {
    evaluating.value = false
  }
}

function formatMetricName(key) {
  return metricLabels[key] || key
}

function formatMetricValue(metricData) {
  if (metricData == null) return '-'
  // 新格式：{value, status, reason}
  if (typeof metricData === 'object') {
    if (metricData.status === 'failed' || metricData.value == null) return '评估失败'
    return (metricData.value * 100).toFixed(2) + '%'
  }
  // 兼容旧格式（直接数值）
  if (isNaN(metricData)) return '-'
  return (metricData * 100).toFixed(2) + '%'
}

function isMetricFailed(metricData) {
  if (metricData == null) return true
  if (typeof metricData === 'object') {
    return metricData.status === 'failed' || metricData.value == null
  }
  return isNaN(metricData)
}

function getMetricReason(metricData) {
  if (metricData == null) return '指标未返回'
  if (typeof metricData === 'object' && metricData.reason) {
    return metricData.reason
  }
  return ''
}

onMounted(() => {
  loadTestQueries()
})
</script>

<template>
  <div class="evaluation-view">
    <div class="page-header">
      <h2>策略评估</h2>
      <p class="subtitle">基于 ragas 框架评估单条查询的检索策略（稠密/稀疏权重、重排序）效果</p>
    </div>

    <!-- 测试集状态 -->
    <el-card class="section-card">
      <template #header>
        <div class="card-header">
          <span>测试集</span>
          <el-tag :type="hasGroundTruth ? 'success' : 'danger'" size="small">
            {{ hasGroundTruth ? '标准答案已就绪' : '标准答案缺失' }}
          </el-tag>
        </div>
      </template>
      <div class="test-set-info">
        <span class="info-item">查询数量：<strong>{{ testQueries.length }}</strong></span>
        <span class="info-item">数据来源：<strong>files/docs/evaluation_qa.md</strong></span>
      </div>
    </el-card>

    <!-- 查询输入 + 策略配置 -->
    <el-card class="section-card">
      <template #header>
        <span>查询输入与策略配置</span>
      </template>

      <!-- 查询输入 -->
      <div class="config-row">
        <label class="config-label">输入查询</label>
        <div class="config-control">
          <el-input
            v-model="queryText"
            placeholder="请输入 files/docs/evaluation_qa.md 中已有标准答案的查询"
            clearable
            maxlength="2000"
            style="width: 100%; max-width: 600px;"
            @keyup.enter="handleEvaluate"
          />
        </div>
      </div>

      <div v-if="queryText.trim() && !selectedEntry" class="query-hint">
        当前查询未匹配到标准答案，请检查输入是否与 files/docs/evaluation_qa.md 中的问题一致。
      </div>

      <div v-if="selectedEntry" class="query-detail">
        <span class="detail-label">相关文档：</span>{{ selectedEntry.related_docs }}
      </div>

      <!-- 策略配置 -->
      <div class="strategy-config">
        <div class="config-row">
          <label class="config-label">稠密向量权重</label>
          <div class="config-control">
            <el-slider
              v-model="denseWeight"
              :min="0"
              :max="1"
              :step="0.1"
              :show-tooltip="true"
              style="flex: 1; max-width: 300px;"
            />
            <span class="weight-display">
              稠密 {{ denseWeight.toFixed(1) }} : 稀疏 {{ sparseWeight.toFixed(1) }}
            </span>
          </div>
        </div>

        <div class="config-row">
          <label class="config-label">检索深度 K</label>
          <div class="config-control">
            <el-input-number v-model="retrievalK" :min="1" :max="20" size="default" />
          </div>
        </div>

        <div class="config-row">
          <label class="config-label">BGE 重排序</label>
          <div class="config-control">
            <el-switch v-model="useRerank" />
            <span class="switch-label">{{ useRerank ? '已启用' : '未启用' }}</span>
          </div>
        </div>

        <div class="config-row" v-if="useRerank">
          <label class="config-label">重排序返回数</label>
          <div class="config-control">
            <el-input-number v-model="rerankTopN" :min="1" :max="10" size="default" />
          </div>
        </div>
      </div>

      <div class="evaluate-action">
        <el-button
          type="primary"
          size="large"
          :loading="evaluating"
          :disabled="evaluating || !hasGroundTruth"
          @click="handleEvaluate"
        >
          {{ evaluating ? '正在评估...' : '开始评估' }}
        </el-button>
        <span v-if="!hasGroundTruth" class="hint-text">标准答案缺失，无法评估</span>
      </div>
    </el-card>

    <!-- 评估结果 -->
    <el-card v-if="evalResult" class="section-card">
      <template #header>
        <span>评估结果</span>
      </template>

      <!-- 查询与策略信息 -->
      <div class="result-header">
        <div class="result-query">
          <strong>查询：</strong>{{ evalResult.query }}
        </div>
        <el-descriptions :column="5" border size="small" style="margin-top: 8px;">
          <el-descriptions-item label="稠密权重">{{ evalResult.strategy.dense_weight }}</el-descriptions-item>
          <el-descriptions-item label="稀疏权重">{{ evalResult.strategy.sparse_weight }}</el-descriptions-item>
          <el-descriptions-item label="重排序">{{ evalResult.strategy.use_rerank ? '启用' : '未启用' }}</el-descriptions-item>
          <el-descriptions-item label="K">{{ evalResult.strategy.k }}</el-descriptions-item>
          <el-descriptions-item label="来源数">{{ (evalResult.sources || []).length }}</el-descriptions-item>
        </el-descriptions>
      </div>

      <!-- 指标卡片 -->
      <div class="metrics-summary">
        <div
          v-for="(metricData, key) in evalResult.metrics"
          :key="key"
          class="metric-card"
          :class="{ 'metric-failed': isMetricFailed(metricData) }"
        >
          <div class="metric-value">{{ formatMetricValue(metricData) }}</div>
          <div class="metric-label">{{ formatMetricName(key) }}</div>
          <el-tooltip
            v-if="(evalResult.fallback_metrics || []).includes(key)"
            :content="evalResult.metric_errors?.[key] || 'Ragas 结构化评估未返回有效值，本项使用本地文本支撑度算法补算'"
            placement="bottom"
          >
            <el-tag type="warning" size="small" class="fail-tag">本地补算</el-tag>
          </el-tooltip>
          <el-tooltip
            v-else-if="isMetricFailed(metricData)"
            :content="getMetricReason(metricData)"
            placement="bottom"
          >
            <el-tag type="danger" size="small" class="fail-tag">失败</el-tag>
          </el-tooltip>
        </div>
      </div>

      <!-- 生成的回答 -->
      <div class="answer-section">
        <h4 class="section-title">🤖 生成的回答</h4>
        <div class="answer-content markdown-body">{{ evalResult.response }}</div>
      </div>

      <!-- 标准答案 -->
      <div class="answer-section">
        <h4 class="section-title">✅ 标准答案</h4>
        <div class="answer-content reference">{{ evalResult.ground_truth }}</div>
      </div>

      <!-- 检索来源 -->
      <div class="answer-section" v-if="evalResult.sources && evalResult.sources.length > 0">
        <h4 class="section-title">📎 检索来源（{{ evalResult.sources.length }}）</h4>
        <el-collapse>
          <el-collapse-item
            v-for="(src, i) in evalResult.sources"
            :key="i"
            :title="`[${i + 1}] ${src.title}（相关度 ${src.score}）`"
            :name="i"
          >
            <div class="source-preview">{{ src.content }}</div>
          </el-collapse-item>
        </el-collapse>
      </div>
    </el-card>

    <!-- 历史结果对比 -->
    <el-card v-if="resultHistory.length > 1" class="section-card">
      <template #header>
        <span>历史结果对比</span>
      </template>
      <el-table :data="resultHistory" stripe style="width: 100%;">
        <el-table-column type="index" label="#" width="50" />
        <el-table-column label="查询" min-width="200" show-overflow-tooltip>
          <template #default="{ row }">{{ row.query }}</template>
        </el-table-column>
        <el-table-column label="稠密:稀疏" width="100">
          <template #default="{ row }">
            {{ row.strategy.dense_weight }} : {{ row.strategy.sparse_weight }}
          </template>
        </el-table-column>
        <el-table-column label="重排序" width="80">
          <template #default="{ row }">
            {{ row.strategy.use_rerank ? '启用' : '关闭' }}
          </template>
        </el-table-column>
        <el-table-column label="K" width="50" prop="strategy.k" />
        <el-table-column
          v-for="(metricData, key) in (resultHistory[0]?.metrics || {})"
          :key="key"
          :label="formatMetricName(key)"
          width="120"
        >
          <template #default="{ row }">
            <span :class="{ 'metric-text-failed': isMetricFailed(row.metrics?.[key]) }">
              {{ formatMetricValue(row.metrics?.[key]) }}
            </span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.evaluation-view {
  max-width: 1000px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0 0 8px 0;
  font-size: 22px;
}

.subtitle {
  color: #909399;
  font-size: 14px;
  margin: 0;
}

.section-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.test-set-info {
  display: flex;
  align-items: center;
  gap: 20px;
}

.info-item {
  font-size: 14px;
  color: #606266;
}


.strategy-config {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin-top: 16px;
}

.config-row {
  display: flex;
  align-items: center;
  gap: 16px;
}

.config-label {
  width: 120px;
  font-size: 14px;
  color: #606266;
  flex-shrink: 0;
}

.config-control {
  display: flex;
  align-items: center;
  gap: 16px;
  flex: 1;
}

.weight-display {
  font-size: 13px;
  color: #409eff;
  font-weight: 500;
  white-space: nowrap;
}

.switch-label {
  font-size: 13px;
  color: #909399;
}

.query-hint {
  margin: 10px 0 0 136px;
  font-size: 13px;
  color: #e6a23c;
}

.query-detail {
  margin: 10px 0 0 136px;
  font-size: 13px;
  color: #606266;
}

.detail-label {
  color: #909399;
}

.evaluate-action {
  margin-top: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.hint-text {
  font-size: 13px;
  color: #e6a23c;
}

.result-header {
  margin-bottom: 20px;
}

.result-query {
  font-size: 14px;
  color: #303133;
  margin-bottom: 8px;
}

.metrics-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 20px;
}

.metric-card {
  flex: 1;
  min-width: 140px;
  padding: 20px;
  background: #f5f7fa;
  border-radius: 8px;
  text-align: center;
}

.metric-value {
  font-size: 24px;
  font-weight: 700;
  color: #409eff;
  margin-bottom: 8px;
}

.metric-label {
  font-size: 13px;
  color: #909399;
}

.metric-card.metric-failed {
  background: #fef0f0;
  border: 1px solid #fde2e2;
}

.metric-card.metric-failed .metric-value {
  color: #f56c6c;
  font-size: 18px;
}

.fail-tag {
  margin-top: 8px;
}

.metric-text-failed {
  color: #f56c6c;
  font-size: 12px;
}

.answer-section {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px dashed #ebeef5;
}

.section-title {
  margin: 0 0 12px 0;
  font-size: 15px;
  color: #303133;
}

.answer-content {
  padding: 16px;
  background: #f5f7fa;
  border-radius: 6px;
  font-size: 14px;
  line-height: 1.7;
  color: #303133;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 400px;
  overflow-y: auto;
}

.answer-content.reference {
  background: #f0f9eb;
  border-left: 3px solid #67c23a;
}

.source-preview {
  font-size: 13px;
  color: #606266;
  line-height: 1.6;
  padding: 8px;
  background: #fafafa;
  border-radius: 4px;
}
</style>
