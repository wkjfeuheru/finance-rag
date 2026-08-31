<script setup>
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Loading } from '@element-plus/icons-vue'
import { complianceDocumentReviewUpload, complianceReview } from '../api'

const inputText = ref('')
const fileInput = ref(null)
const loading = ref(false)
const result = ref(null)
const mode = ref('text')

const reportSuggestions = computed(() => (result.value?.suggestions || []).map((suggestion, index) => ({
  id: `S-${String(index + 1).padStart(2, '0')}`,
  text: typeof suggestion === 'string' ? suggestion : suggestion.text || suggestion.content || ''
})).filter(suggestion => suggestion.text))

// 结构化法规证据优先，兼容旧接口返回的字符串证据。
const reportEvidence = computed(() => {
  if (result.value?.regulation_evidence?.length) return result.value.regulation_evidence
  return (result.value?.evidence || []).map((item, index) => ({
    evidence_id: `E-${String(index + 1).padStart(2, '0')}`,
    title: '检索证据',
    quote: typeof item === 'string' ? item : item.quote || item.content || '',
    clause: '',
    validation_status: ''
  }))
})

const riskLabels = {
  safe: '安全 / 放行',
  controversial: '存在争议 / 人工审核',
  unsafe: '不安全 / 拦截'
}

const actionLabels = {
  pass: '放行',
  manual_review: '转人工审核',
  reject: '拦截'
}

const riskType = computed(() => {
  if (!result.value?.risk_level) return 'info'
  return {
    safe: 'success',
    controversial: 'warning',
    unsafe: 'danger'
  }[result.value.risk_level] || 'info'
})

function pickFile() {
  if (!loading.value) fileInput.value?.click()
}

function onFileChange(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (file) reviewFile(file)
}

async function reviewText() {
  const content = inputText.value.trim()
  if (!content || loading.value) return

  loading.value = true
  result.value = null
  try {
    result.value = await complianceReview({
      query: content,
      history: [],
      use_rerank: true
    })
  } catch (error) {
    ElMessage.error(`审查失败：${error.response?.data?.detail || error.message}`)
  } finally {
    loading.value = false
  }
}

async function reviewFile(file) {
  loading.value = true
  result.value = null
  try {
    result.value = await complianceDocumentReviewUpload(file)
    mode.value = 'file'
  } catch (error) {
    ElMessage.error(`文档审查失败：${error.response?.data?.detail || error.message}`)
  } finally {
    loading.value = false
  }
}

function clearReview() {
  if (loading.value) return
  inputText.value = ''
  result.value = null
  mode.value = 'text'
}
</script>

<template>
  <div class="compliance-page">
    <div class="page-header">
      <div>
        <h2 class="page-title">合规审查</h2>
        <p class="page-subtitle">输入业务描述或上传待审文档，获取法规依据和处置建议</p>
      </div>
      <el-button :disabled="loading || (!inputText && !result)" @click="clearReview">清空</el-button>
    </div>

    <section class="review-input card">
      <div class="input-heading">
        <div>
          <h3>提交审查内容</h3>
          <span>文本内容将通过独立合规审查 Agent 处理</span>
        </div>
        <el-button plain :loading="loading" @click="pickFile">上传文档</el-button>
        <input
          ref="fileInput"
          type="file"
          accept=".md,.txt,.pdf,.docx,.markdown"
          class="hidden-input"
          @change="onFileChange"
        />
      </div>
      <el-input
        v-model="inputText"
        type="textarea"
        :rows="9"
        maxlength="30000"
        show-word-limit
        resize="vertical"
        :disabled="loading"
        placeholder="例如：描述一项金融产品、营销文案、业务流程或待审核条款..."
      />
      <div class="input-footer">
        <span class="input-hint">支持长文本审查，系统会自动识别红线并检索相关法规</span>
        <el-button type="primary" :loading="loading" :disabled="!inputText.trim()" @click="reviewText">
          开始审查
        </el-button>
      </div>
    </section>

    <section v-if="loading" class="loading-state card">
      <el-icon class="is-loading"><Loading /></el-icon>
      <span>正在识别风险并检索法规证据...</span>
    </section>

    <section v-if="result" class="review-result">
      <div class="result-summary card">
        <div class="result-heading">
          <div>
            <span class="eyebrow">审查结果</span>
            <h3>{{ mode === 'file' ? '文档合规审查报告' : '业务内容合规审查报告' }}</h3>
          </div>
          <el-tag :type="riskType" size="large">
            {{ riskLabels[result.risk_level] || '待确认' }}
          </el-tag>
        </div>
        <div class="summary-grid">
          <div>
            <span class="summary-label">处置动作</span>
            <strong>{{ actionLabels[result.action] || result.action || '待确认' }}</strong>
          </div>
          <div>
            <span class="summary-label">风险等级</span>
            <strong>{{ result.risk_level || '待确认' }}</strong>
          </div>
        </div>
        <div class="report-block">
          <h4>审核理由</h4>
          <p>{{ result.reason || result.answer || result.assessment || '暂无理由' }}</p>
        </div>
        <div v-if="reportSuggestions.length" class="report-block">
          <h4>修改建议 <span>{{ reportSuggestions.length }}</span></h4>
          <ul class="suggestion-list">
            <li v-for="suggestion in reportSuggestions" :key="suggestion.id" class="suggestion-item">
              <span class="suggestion-id">{{ suggestion.id }}</span>
              <span class="suggestion-text">{{ suggestion.text }}</span>
            </li>
          </ul>
        </div>
      </div>

      <div v-if="result.findings?.length" class="findings-section">
        <div class="section-heading">
          <div><span class="eyebrow">Evidence Traceability</span><h3>逐项审查与证据溯源</h3></div>
          <el-tag :type="result.evidence_coverage >= 1 ? 'success' : 'warning'" size="small">
            证据完整率 {{ Math.round((result.evidence_coverage || 0) * 100) }}%
          </el-tag>
        </div>
        <article v-for="finding in result.findings" :key="finding.finding_id" class="finding-card">
          <div class="finding-heading">
            <div><el-tag :type="finding.risk_level === 'unsafe' ? 'danger' : 'warning'" size="small">{{ finding.finding_id }}</el-tag> <strong>{{ finding.summary }}</strong></div>
            <el-tag :type="finding.evidence_status === 'verified' ? 'success' : 'warning'" size="small">
              {{ finding.evidence_status === 'verified' ? '证据已验证' : '证据不足，需人工审核' }}
            </el-tag>
          </div>
          <p class="finding-reason">{{ finding.reason }}</p>
          <div class="evidence-grid">
            <div class="evidence-pane"><h4>待审原文</h4>
              <blockquote v-for="item in finding.document_evidence" :key="item.location.segment_id">
                {{ item.location.quote }}<small>{{ item.location.page ? `第 ${item.location.page} 页 · ` : '' }}{{ item.location.line_start ? `第 ${item.location.line_start} 行` : '字符位置可追溯' }}</small>
              </blockquote>
            </div>
            <div class="evidence-pane"><h4>法规依据</h4>
              <div v-if="finding.regulation_evidence?.length">
                <div v-for="evidence in finding.regulation_evidence" :key="evidence.evidence_id" class="regulation-evidence">
                  <div class="evidence-meta"><el-tag size="small" effect="plain">{{ evidence.evidence_id }}</el-tag> <span>{{ evidence.title || evidence.source }}</span></div>
                  <p>{{ evidence.quote }}</p><small>{{ evidence.clause || '未定位具体条款' }} · {{ evidence.validation_status }}</small>
                </div>
              </div><p v-else class="muted">未绑定法规证据</p>
            </div>
          </div>
          <ol v-if="finding.suggestions?.length" class="finding-suggestions">
            <li v-for="(suggestion, index) in finding.suggestions" :key="suggestion" class="suggestion-item">
              <span class="suggestion-id">S-{{ String(index + 1).padStart(2, '0') }}</span>
              <span class="suggestion-text">{{ suggestion }}</span>
            </li>
          </ol>
        </article>
      </div>

      <div class="result-columns">
        <div class="card report-block">
          <h4>命中红线 <span>{{ result.red_lines?.length || 0 }}</span></h4>
          <div v-if="result.red_lines?.length" class="redline-list">
            <div v-for="line in result.red_lines" :key="line.rule_id" class="redline-item">
              <el-tag type="danger" size="small">{{ line.rule_id }}</el-tag>
              <div>{{ line.behavior }}<small>{{ line.legal_basis }}</small></div>
            </div>
          </div>
          <p v-else class="muted">未命中预设红线规则</p>
        </div>

        <div class="card report-block">
          <h4>引用证据 <span>{{ reportEvidence.length }}</span></h4>
          <div v-if="reportEvidence.length" class="report-evidence-list">
            <article v-for="(evidence, index) in reportEvidence" :key="evidence.evidence_id || index" class="report-evidence-item">
              <div class="evidence-meta">
                <el-tag size="small" effect="plain">{{ evidence.evidence_id || `E-${String(index + 1).padStart(2, '0')}` }}</el-tag>
                <strong>{{ evidence.title || evidence.source || '法规证据' }}</strong>
              </div>
              <p v-if="evidence.clause" class="evidence-clause">{{ evidence.clause }}</p>
              <blockquote v-if="evidence.quote">{{ evidence.quote }}</blockquote>
              <small v-if="evidence.validation_status" class="evidence-status">{{ evidence.validation_status }}</small>
            </article>
          </div>
          <p v-else class="muted">暂无结构化证据</p>
        </div>
      </div>

      <div v-if="result.sources?.length" class="card report-block sources-block">
        <h4>法规来源 <span>{{ result.sources.length }}</span></h4>
        <el-collapse>
          <el-collapse-item v-for="source in result.sources" :key="source.index" :name="source.index" :title="`[${source.index}] ${source.title}`">
            <p class="source-preview">{{ source.preview }}</p>
          </el-collapse-item>
        </el-collapse>
      </div>
    </section>
  </div>
</template>

<style scoped>
.compliance-page { max-width: 1000px; margin: 0 auto; padding-bottom: 24px; }
.page-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 18px; }
.page-title { margin: 0 0 6px; }
.page-subtitle { margin: 0; color: var(--text-muted); font-size: 13px; }
.card { background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-md); }
.review-input { padding: 18px; }
.input-heading, .input-footer, .result-heading { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.input-heading { margin-bottom: 12px; }
.input-heading h3, .result-heading h3 { margin: 0 0 4px; font-size: 16px; }
.input-heading span { color: var(--text-muted); font-size: 12px; }
.hidden-input { display: none; }
.input-footer { margin-top: 12px; }
.input-hint, .muted, .source-preview { color: var(--text-muted); font-size: 12px; }
.loading-state { display: flex; align-items: center; gap: 8px; margin-top: 14px; padding: 18px; color: var(--accent); font-size: 13px; }
.review-result { margin-top: 14px; }
.result-summary { padding: 20px; }
.eyebrow, .summary-label { display: block; color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
.summary-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }
.summary-grid > div { background: var(--bg-elevated); padding: 12px; border-radius: 4px; }
.summary-grid strong { display: block; margin-top: 5px; font-size: 14px; }
.report-block { padding: 16px; }
.report-block h4 { margin: 0 0 10px; font-size: 14px; }
.report-block h4 span { color: var(--text-muted); font-weight: normal; }
.report-block p, .report-block li { color: var(--text-secondary); font-size: 13px; line-height: 1.7; }
.report-block p { margin: 0; }
.report-block ul { margin: 0; padding-left: 18px; }
.findings-section { margin-top: 14px; }
.section-heading, .finding-heading, .evidence-meta { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.section-heading { margin: 0 2px 10px; }
.section-heading h3 { margin: 3px 0 0; font-size: 16px; }
.finding-card { background: var(--bg-surface); border: 1px solid var(--border); border-left: 3px solid var(--accent); padding: 16px; margin-bottom: 10px; }
.finding-heading strong { margin-left: 6px; font-size: 14px; }
.finding-reason { color: var(--text-secondary); font-size: 13px; line-height: 1.6; margin: 12px 0; }
.evidence-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.evidence-pane { background: var(--bg-elevated); padding: 12px; min-width: 0; }
.evidence-pane h4 { margin: 0 0 8px; font-size: 13px; }
.evidence-pane blockquote { margin: 0; padding: 9px 10px; border-left: 2px solid var(--accent); color: var(--text-secondary); font-size: 13px; line-height: 1.6; }
.evidence-pane small, .regulation-evidence small { display: block; color: var(--text-muted); font-size: 11px; margin-top: 5px; }
.regulation-evidence { border-top: 1px solid var(--border); padding: 8px 0; }
.regulation-evidence:first-child { border-top: 0; padding-top: 0; }
.regulation-evidence p { margin: 6px 0 0; font-size: 12px; line-height: 1.6; color: var(--text-secondary); }
.suggestion-list, .finding-suggestions { margin: 0; padding: 0; list-style: none; display: grid; gap: 8px; }
.finding-suggestions { margin-top: 12px; }
.suggestion-item { display: flex; align-items: flex-start; gap: 8px; }
.suggestion-id { flex: 0 0 auto; color: var(--accent); font-size: 12px; font-weight: 600; line-height: 1.7; }
.suggestion-text { min-width: 0; }
.report-evidence-list { display: grid; gap: 10px; }
.report-evidence-item { min-width: 0; padding: 10px; border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 4px; background: var(--bg-elevated); }
.report-evidence-item .evidence-meta { justify-content: flex-start; align-items: flex-start; }
.report-evidence-item .evidence-meta strong { min-width: 0; color: var(--text-secondary); font-size: 12px; line-height: 1.5; overflow-wrap: anywhere; }
.evidence-clause { margin: 7px 0 0; color: var(--text-muted); font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
.report-evidence-item blockquote { margin: 7px 0 0; padding: 8px 9px; border-left: 2px solid var(--accent); color: var(--text-secondary); font-size: 12px; line-height: 1.6; overflow-wrap: anywhere; }
.evidence-status { display: block; margin-top: 6px; color: var(--text-muted); font-size: 11px; overflow-wrap: anywhere; }
.redline-item { display: flex; gap: 8px; padding: 8px 0; border-top: 1px solid var(--border); color: var(--text-secondary); font-size: 13px; line-height: 1.5; }
.redline-item small { display: block; color: var(--text-muted); margin-top: 2px; }
.sources-block { margin-top: 14px; }
.source-preview { line-height: 1.6; margin: 0; }
@media (max-width: 700px) {
  .page-header, .input-footer, .result-heading { align-items: stretch; flex-direction: column; }
  .result-heading .el-tag { align-self: flex-start; }
  .result-columns, .evidence-grid { grid-template-columns: 1fr; }
  .finding-heading { align-items: flex-start; flex-direction: column; }
  .input-footer .el-button { width: 100%; }
}
</style>
