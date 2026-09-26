<script setup>
import { ref, reactive, nextTick, onUnmounted } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { Loading } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useChatStore } from '../store/chat'
import { chatStream, fetchDocumentPage } from '../api'
import { categoryLabel } from '../constants/categories'

const store = useChatStore()

const inputText = ref('')
const scrollContainer = ref(null)
let abortController = null

// 关闭后不再让后端用 LLM 推断过滤条件（清除 chip 后靠它生效）
const suppressAutoFilter = ref(false)

// 原文页弹层
const pageDialog = reactive({
  visible: false,
  title: '',
  page: 0,
  url: '',
  loading: false
})

const stageLabels = {
  rewriting: '正在改写查询...',
  inferring_filters: '正在分析过滤条件...',
  retrieving: '正在检索知识库...',
  hyde: '正在生成假设回答...',
  reranking: '正在重排序...',
  generating: '正在生成回答...'
}

const blockTypeLabels = { table: '表格', image: '图片', text: '' }

// 将后端推断出的过滤条件转为可读 chip 列表
function filterChips(filters) {
  if (!filters) return []
  const chips = []
  if (filters.category?.length) {
    chips.push({ key: 'category', text: `分类：${filters.category.map(categoryLabel).join('/')}` })
  }
  if (filters.security_code?.length) {
    chips.push({ key: 'security_code', text: `标的：${filters.security_code.join('/')}` })
  }
  if (filters.industry_l1?.length) {
    chips.push({ key: 'industry_l1', text: `行业：${filters.industry_l1.join('/')}` })
  }
  if (filters.industry_l2?.length) {
    chips.push({ key: 'industry_l2', text: `子行业：${filters.industry_l2.join('/')}` })
  }
  if (filters.report_type?.length) {
    chips.push({ key: 'report_type', text: `类型：${filters.report_type.join('/')}` })
  }
  if (filters.broker?.length) {
    chips.push({ key: 'broker', text: `机构：${filters.broker.join('/')}` })
  }
  if (filters.date?.gte) chips.push({ key: 'date_gte', text: `不早于 ${filters.date.gte}` })
  if (filters.date?.lte) chips.push({ key: 'date_lte', text: `不晚于 ${filters.date.lte}` })
  return chips
}

function clearAutoFilter() {
  suppressAutoFilter.value = true
  ElMessage.info('已关闭自动过滤，下次提问将检索全部研报')
}

function restoreAutoFilter() {
  suppressAutoFilter.value = false
  ElMessage.info('已恢复自动过滤')
}

async function scrollToBottom() {
  await nextTick()
  if (scrollContainer.value) {
    scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight
  }
}

function renderMarkdown(text) {
  try {
    return DOMPurify.sanitize(marked(text))
  } catch {
    return text
  }
}

function pageLabel(src) {
  if (!src.start_page) return ''
  if (!src.end_page || src.end_page === src.start_page) return `第 ${src.start_page} 页`
  return `第 ${src.start_page}-${src.end_page} 页`
}

// 打开原文页：结论验证的关键一步——拿到页码直接看到那一页
async function openSourcePage(src) {
  const page = src.start_page || 1
  pageDialog.title = src.title || src.source
  pageDialog.page = page
  pageDialog.loading = true
  pageDialog.visible = true
  if (pageDialog.url) {
    URL.revokeObjectURL(pageDialog.url)
    pageDialog.url = ''
  }
  try {
    pageDialog.url = await fetchDocumentPage(src.source, page)
  } catch (err) {
    pageDialog.visible = false
    ElMessage.error(`无法打开原文页：${err.message}`)
  } finally {
    pageDialog.loading = false
  }
}

async function sendMessage() {
  const query = inputText.value.trim()
  if (!query || store.streaming) return

  // 添加用户消息
  store.addMessage({ role: 'user', content: query, time: new Date().toLocaleTimeString() })

  // 添加占位助手消息
  store.addMessage({
    role: 'assistant',
    content: '',
    sources: [],
    stage: '',
    streaming: true,
    time: new Date().toLocaleTimeString()
  })

  inputText.value = ''
  store.setStreaming(true)

  // 构建历史（排除当前占位消息）
  const history = store.messages
    .slice(0, -2)
    .map(m => ({ role: m.role, content: m.content }))
    .filter(m => m.content)

  // 过滤条件由后端 LLM 根据问题自动推断（Self-querying）；
  // 分析师点掉 chip 后置 infer_filters=false，避免又被推断回来。
  // k/rerank_top_n 交由后端默认值/动态 K 决定
  const payload = {
    query,
    history,
    use_rerank: true,
    infer_filters: !suppressAutoFilter.value
  }

  abortController = chatStream(
    payload,
    (event) => {
      store.updateLastAssistant((msg) => {
        if (event.type === 'status') {
          msg.stage = event.stage
        } else if (event.type === 'sources') {
          msg.sources = event.sources || []
          msg.stage = ''
        } else if (event.type === 'token') {
          msg.content += event.content
          msg.stage = ''
          scrollToBottom()
        } else if (event.type === 'done') {
          msg.streaming = false
          msg.stage = ''
          msg.rewrittenQuery = event.rewritten_query
          msg.appliedFilters = event.filters || null
          // 幻觉治理状态：此前前端完全没有读取，可信与否在界面上不可见
          msg.citationValidation = event.citation_validation || null
          msg.answerRejected = !!event.answer_rejected
          msg.lowConfidence = !!event.low_confidence
        } else if (event.type === 'error') {
          msg.content = `❌ 错误：${event.message}`
          msg.streaming = false
          msg.stage = ''
        }
      })
      // error 事件也需要重置 streaming 状态
      if (event.type === 'error' || event.type === 'done') {
        store.setStreaming(false)
        abortController = null
      }
    },
    (err) => {
      store.updateLastAssistant((msg) => {
        msg.content = `❌ 请求失败：${err.message}`
        msg.streaming = false
        msg.stage = ''
      })
      store.setStreaming(false)
      abortController = null
    }
  )
}

function stopStreaming() {
  if (abortController) {
    abortController.abort()
    abortController = null
  }
  store.updateLastAssistant((msg) => {
    msg.streaming = false
    msg.stage = ''
  })
  store.setStreaming(false)
}

function clearChat() {
  if (store.streaming) return
  store.clearMessages()
}

onUnmounted(() => {
  if (abortController) abortController.abort()
  if (pageDialog.url) URL.revokeObjectURL(pageDialog.url)
})
</script>

<template>
  <div class="chat-page">
    <div class="chat-header">
      <h2 class="page-title">智能问答</h2>
      <el-button size="small" @click="clearChat" :disabled="store.streaming">清空对话</el-button>
    </div>

    <div class="chat-body" ref="scrollContainer">
      <div v-if="store.messages.length === 0" class="empty-state">
        <div class="empty-icon">💬</div>
        <p>支持提问各种金融领域问题，获取专业回答</p>
      </div>

      <div
        v-for="(msg, idx) in store.messages"
        :key="idx"
        class="message"
        :class="msg.role"
      >
        <div class="message-avatar">
          {{ msg.role === 'user' ? '👤' : '🤖' }}
        </div>
        <div class="message-body">
          <div class="message-meta">
            <span>{{ msg.role === 'user' ? '我' : '助手' }}</span>
            <span class="message-time">{{ msg.time }}</span>
            <span v-if="msg.rewrittenQuery && msg.rewrittenQuery !== store.messages[idx-1]?.content" class="rewrite-tag">
              改写查询：{{ msg.rewrittenQuery }}
            </span>
          </div>

          <!-- 自动推断的过滤条件：可见、可清除 -->
          <div v-if="msg.role === 'assistant' && filterChips(msg.appliedFilters).length" class="filter-chips">
            <span class="chips-label">已按以下条件收窄检索：</span>
            <el-tag
              v-for="chip in filterChips(msg.appliedFilters)"
              :key="chip.key"
              size="small"
              type="info"
              class="filter-chip"
            >
              {{ chip.text }}
            </el-tag>
            <el-button link type="primary" size="small" @click="clearAutoFilter">
              清除过滤（下次提问生效）
            </el-button>
          </div>

          <!-- 幻觉治理状态：拒答 / 低置信 / 引用校验 -->
          <div v-if="msg.role === 'assistant' && (msg.answerRejected || msg.lowConfidence || msg.citationValidation)" class="credibility-row">
            <el-tag v-if="msg.answerRejected" type="danger" size="small">已拒答：检索证据不足</el-tag>
            <el-tag v-else-if="msg.lowConfidence" type="warning" size="small">
              低置信：引用校验未达标，请核对原文
            </el-tag>
            <span v-if="msg.citationValidation" class="citation-score">
              引用校验：{{ msg.citationValidation.verified_citations ?? 0 }}/{{ msg.citationValidation.total_citations ?? 0 }} 条通过（得分 {{ msg.citationValidation.score ?? '—' }}）
            </span>
          </div>

          <!-- 阶段状态 -->
          <div v-if="msg.stage" class="stage-indicator">
            <el-icon class="is-loading"><Loading /></el-icon>
            <span>{{ stageLabels[msg.stage] || msg.stage }}</span>
          </div>

          <!-- 消息内容 -->
          <div
            v-if="msg.content"
            class="message-content markdown-body"
            v-html="renderMarkdown(msg.content)"
          ></div>

          <!-- 来源引用 -->
          <div v-if="msg.sources && msg.sources.length > 0" class="sources-panel">
            <div class="sources-title">
              📎 引用来源（{{ msg.sources.length }}）
              <el-button
                v-if="suppressAutoFilter"
                link
                type="primary"
                size="small"
                @click="restoreAutoFilter"
              >
                恢复自动过滤
              </el-button>
            </div>
            <el-collapse>
              <el-collapse-item
                v-for="src in msg.sources"
                :key="src.index"
                :name="src.index"
              >
                <template #title>
                  <span class="source-title">
                    [{{ src.index }}] {{ src.title }}（相关度 {{ src.score }}）
                    <el-tag v-if="src.date" size="small" class="source-tag">{{ src.date }}</el-tag>
                    <el-tag v-if="blockTypeLabels[src.block_type]" size="small" type="warning" class="source-tag">
                      {{ blockTypeLabels[src.block_type] }}
                    </el-tag>
                    <el-tag v-if="src.truncated" size="small" type="danger" class="source-tag">表已截断</el-tag>
                  </span>
                </template>
                <div class="source-preview">{{ src.preview }}</div>
                <div class="source-actions">
                  <span v-if="pageLabel(src)" class="source-page">{{ pageLabel(src) }}</span>
                  <el-button
                    v-if="src.start_page"
                    size="small"
                    type="primary"
                    plain
                    @click.stop="openSourcePage(src)"
                  >
                    查看原文页
                  </el-button>
                </div>
              </el-collapse-item>
            </el-collapse>
          </div>
        </div>
      </div>
    </div>

    <!-- 原文页弹层：页码级溯源的落点 -->
    <el-dialog v-model="pageDialog.visible" :title="`${pageDialog.title} — 第 ${pageDialog.page} 页`" width="820px">
      <div v-loading="pageDialog.loading" class="page-viewer">
        <img v-if="pageDialog.url" :src="pageDialog.url" alt="原文页" />
      </div>
    </el-dialog>

    <div class="chat-input-area">
      <el-input
        v-model="inputText"
        type="textarea"
        :rows="2"
        placeholder="输入您的金融问题，按 Enter 发送，Shift+Enter 换行"
        :disabled="store.streaming"
        @keydown.enter.exact.prevent="sendMessage"
      />
      <div class="input-actions">
        <el-button
          v-if="!store.streaming"
          type="primary"
          @click="sendMessage"
          :disabled="!inputText.trim()"
        >
          发送
        </el-button>
        <el-button
          v-else
          type="danger"
          @click="stopStreaming"
        >
          停止生成
        </el-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  max-width: 860px;
  margin: 0 auto;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}

.chat-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
}

.empty-state {
  text-align: center;
  padding: 80px 20px;
  color: var(--text-muted);
}
.empty-icon { font-size: 40px; margin-bottom: 14px; opacity: 0.6; }
.empty-state p { font-size: 14px; }

.message {
  display: flex;
  gap: 12px;
  margin-bottom: 22px;
}
.message-avatar {
  font-size: 26px;
  flex-shrink: 0;
  width: 34px;
  height: 34px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.message-body { flex: 1; min-width: 0; }
.message-meta {
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 4px;
  display: flex;
  gap: 8px;
  align-items: center;
}
.message-time { font-size: 11px; }
.rewrite-tag {
  background: var(--accent-soft);
  color: var(--accent);
  padding: 1px 8px;
  border-radius: 3px;
  font-size: 11px;
}

.message.user .message-content {
  background: var(--bg-elevated);
  color: var(--text-primary);
  padding: 10px 14px;
  border-radius: var(--radius-md);
  display: inline-block;
  max-width: 80%;
}
.message.assistant .message-content {
  padding: 2px 0;
}

.stage-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--accent);
  font-size: 13px;
  padding: 6px 0;
}

.sources-panel {
  margin-top: 12px;
  border-top: 1px solid var(--border);
  padding-top: 10px;
}
.sources-title {
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.source-title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.source-tag { flex-shrink: 0; }
.source-preview {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.6;
  max-height: 120px;
  overflow-y: auto;
}
.source-actions {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.source-page {
  font-size: 12px;
  color: var(--text-muted);
}

.filter-chips {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  margin: 4px 0 6px;
}
.chips-label {
  font-size: 12px;
  color: var(--text-muted);
}
.filter-chip { font-size: 11px; }

.credibility-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin: 4px 0 6px;
}
.citation-score {
  font-size: 12px;
  color: var(--text-muted);
}

.page-viewer {
  min-height: 240px;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  max-height: 70vh;
  overflow-y: auto;
}
.page-viewer img {
  max-width: 100%;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}

.chat-input-area {
  margin-top: 14px;
  display: flex;
  gap: 10px;
  align-items: flex-end;
}
.chat-input-area :deep(.el-textarea) { flex: 1; }
.input-actions { flex-shrink: 0; }
</style>
