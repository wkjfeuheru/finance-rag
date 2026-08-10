<script setup>
import { ref, nextTick, onUnmounted } from 'vue'
import { marked } from 'marked'
import { Loading } from '@element-plus/icons-vue'
import { useChatStore } from '../store/chat'
import { chatStream } from '../api'

const store = useChatStore()

const inputText = ref('')
const scrollContainer = ref(null)
let abortController = null

const stageLabels = {
  rewriting: '正在改写查询...',
  retrieving: '正在检索知识库...',
  reranking: '正在重排序...',
  generating: '正在生成回答...'
}

async function scrollToBottom() {
  await nextTick()
  if (scrollContainer.value) {
    scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight
  }
}

function renderMarkdown(text) {
  try {
    return marked(text)
  } catch {
    return text
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

  abortController = chatStream(
    { query, history, use_rewrite: true, use_rerank: true, k: 5, rerank_top_n: 3 },
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

  // 监听流结束（通过轮询最后消息状态）
  const checkEnd = setInterval(() => {
    const last = store.messages[store.messages.length - 1]
    if (last && last.role === 'assistant' && !last.streaming) {
      clearInterval(checkEnd)
      store.setStreaming(false)
      abortController = null
      scrollToBottom()
    }
  }, 200)
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
            <div class="sources-title">📎 引用来源（{{ msg.sources.length }}）</div>
            <el-collapse>
              <el-collapse-item
                v-for="src in msg.sources"
                :key="src.index"
                :title="`[${src.index}] ${src.title}（相关度 ${src.score}）`"
                :name="src.index"
              >
                <div class="source-preview">{{ src.preview }}</div>
              </el-collapse-item>
            </el-collapse>
          </div>
        </div>
      </div>
    </div>

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
}
.source-preview {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.6;
  max-height: 120px;
  overflow-y: auto;
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
