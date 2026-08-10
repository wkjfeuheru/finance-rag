<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import { listDocuments, deleteDocument, uploadDocuments, uploadDocumentsAsync, getTaskStatus, getKbStats } from '../api'

const documents = ref([])
const stats = ref({ document_count: 0, chunk_count: 0, exists: false })
const uploading = ref(false)
const uploadProgress = ref(0)
const uploadFiles = ref([])

// 异步上传任务状态
const asyncTasks = ref([]) // [{task_id, filename, status, result, error}]
let _pollTimer = null

async function loadDocuments() {
  try {
    const [docs, st] = await Promise.all([listDocuments(), getKbStats()])
    documents.value = docs
    stats.value = st
  } catch (e) {
    ElMessage.error('加载文档列表失败：' + e.message)
  }
}

function handleFileChange(_file, files) {
  uploadFiles.value = files
}

function handleFileRemove(_file, files) {
  uploadFiles.value = files
}

function hasPdfFiles(files) {
  return files.some(f => f.name.toLowerCase().endsWith('.pdf'))
}

async function handleUpload() {
  const files = uploadFiles.value.map(item => item.raw).filter(Boolean)
  if (!files.length) {
    ElMessage.warning('Please select at least one file')
    return
  }

  // PDF 文件走异步上传，其他走同步
  const pdfFiles = files.filter(f => f.name.toLowerCase().endsWith('.pdf'))
  const otherFiles = files.filter(f => !f.name.toLowerCase().endsWith('.pdf'))

  uploading.value = true
  uploadProgress.value = 0

  try {
    // 同步上传非 PDF 文件
    if (otherFiles.length) {
      const result = await uploadDocuments(otherFiles, (p) => {
        uploadProgress.value = p
      })
      if (result.failure_count) {
        const details = result.failures.map(item => `${item.filename}：${item.error}`).join('；')
        ElMessage.error(`上传失败 ${result.failure_count} 个：${details}`)
      }
      if (result.success_count) {
        ElMessage.success(`非 PDF 文件上传完成：成功 ${result.success_count} 个`)
      }
    }

    // 异步上传 PDF 文件
    if (pdfFiles.length) {
      uploadProgress.value = 50
      const tasks = await uploadDocumentsAsync(pdfFiles, (p) => {
        uploadProgress.value = 50 + Math.round(p * 0.5)
      })
      for (const t of tasks) {
        asyncTasks.value.push({
          task_id: t.task_id,
          filename: t.filename,
          status: 'processing',
          result: null,
          error: null,
        })
      }
      ElMessage.info(`${pdfFiles.length} 个 PDF 已提交后台解析，请等待处理完成`)
      startPolling()
    }

    uploadFiles.value = []
    await loadDocuments()
  } catch (e) {
    const msg = e.code === 'ECONNABORTED'
      ? '处理超时，请检查 Milvus 和嵌入服务'
      : (e.response?.data?.detail || e.message)
    ElMessage.error('上传失败：' + msg)
  } finally {
    uploading.value = false
    uploadProgress.value = 0
  }
}

// ---- 异步任务轮询 ----

function startPolling() {
  if (_pollTimer) return
  _pollTimer = setInterval(pollTasks, 2000)
  pollTasks()
}

function stopPolling() {
  if (_pollTimer) {
    clearInterval(_pollTimer)
    _pollTimer = null
  }
}

async function pollTasks() {
  const pending = asyncTasks.value.filter(t => t.status === 'processing')
  if (!pending.length) {
    stopPolling()
    return
  }
  for (const task of pending) {
    try {
      const data = await getTaskStatus(task.task_id)
      task.status = data.status
      if (data.status === 'completed') {
        task.result = data.result
        ElMessage.success(`PDF 处理完成：${task.filename}（${data.result.chunk_count} 个切块）`)
        await loadDocuments()
      } else if (data.status === 'failed') {
        task.error = data.error
        ElMessage.error(`PDF 处理失败：${task.filename}：${data.error}`)
      }
    } catch {
      // 轮询失败静默忽略
    }
  }
  // 1 分钟后清理已完成任务
  const now = Date.now()
  asyncTasks.value = asyncTasks.value.filter(
    t => t.status === 'processing' || (now - 0) < 60000
  )
}

onUnmounted(() => {
  stopPolling()
})

async function handleDelete(doc) {
  try {
    await ElMessageBox.confirm(
      `确认删除文档「${doc.title}」？此操作将同时删除向量索引。`,
      '删除确认',
      { type: 'warning' }
    )
    await deleteDocument(doc.source)
    ElMessage.success('删除成功')
    await loadDocuments()
  } catch (e) {
    if (e !== 'cancel') {
      ElMessage.error('删除失败：' + (e.message || e))
    }
  }
}

function formatSource(source) {
  // 只显示文件名
  const parts = source.replace(/\\/g, '/').split('/')
  return parts[parts.length - 1]
}

onMounted(() => {
  loadDocuments()
})
</script>

<template>
  <div class="docs-page">
    <h2 class="page-title">文档管理</h2>

    <!-- 统计卡片 -->
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-value">{{ stats.document_count }}</div>
        <div class="stat-label">文档总数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ stats.chunk_count }}</div>
        <div class="stat-label">向量块数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ stats.exists ? '✅' : '❌' }}</div>
        <div class="stat-label">知识库状态</div>
      </div>
    </div>

    <!-- 上传区域 -->
    <div class="card upload-section">
      <el-upload
        drag
        multiple
        :auto-upload="false"
        :file-list="uploadFiles"
        :on-change="handleFileChange"
        :on-remove="handleFileRemove"
        accept=".md,.txt,.pdf"
        :disabled="uploading"
      >
        <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
        <div class="el-upload__text">
          拖拽多个文件到此处，或<em>点击选择文件</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            支持批量上传 .md / .txt / .pdf 格式，单文件不超过 20MB
          </div>
        </template>
      </el-upload>
      <el-button
        type="primary"
        :loading="uploading"
        :disabled="uploading || uploadFiles.length === 0"
        style="margin-top: 12px"
        @click="handleUpload"
      >上传所选文件（{{ uploadFiles.length }}）</el-button>
      <el-progress
        v-if="uploading"
        :percentage="uploadProgress"
        :stroke-width="6"
        style="margin-top: 12px"
      />
    </div>

    <!-- 异步任务状态 -->
    <div v-if="asyncTasks.length" class="card" style="margin-top: 12px">
      <div class="card-header">
        <span>后台处理任务</span>
        <el-button size="small" @click="asyncTasks = asyncTasks.filter(t => t.status === 'processing')">清除已完成</el-button>
      </div>
      <div v-for="task in asyncTasks" :key="task.task_id" class="task-item">
        <span class="task-filename">{{ task.filename }}</span>
        <el-tag
          :type="task.status === 'completed' ? 'success' : task.status === 'failed' ? 'danger' : 'warning'"
          size="small"
        >
          {{ task.status === 'processing' ? '处理中...' : task.status === 'completed' ? '完成' : '失败' }}
        </el-tag>
        <span v-if="task.status === 'completed' && task.result" class="task-detail">
          {{ task.result.chunk_count }} 个切块
        </span>
        <span v-if="task.status === 'failed' && task.error" class="task-error">
          {{ task.error }}
        </span>
      </div>
    </div>

    <!-- 文档列表 -->
    <div class="card" style="margin-top: 16px">
      <div class="card-header">
        <span>知识库文档</span>
        <el-button size="small" @click="loadDocuments">刷新</el-button>
      </div>
      <el-table :data="documents" style="width: 100%" empty-text="暂无文档，请上传">
        <el-table-column prop="title" label="文档标题" min-width="200" />
        <el-table-column label="文件名" min-width="200">
          <template #default="{ row }">{{ formatSource(row.source) }}</template>
        </el-table-column>
        <el-table-column prop="chunk_count" label="向量块数" width="120" align="center" />
        <el-table-column label="操作" width="100" align="center">
          <template #default="{ row }">
            <el-button
              type="danger"
              size="small"
              link
              @click="handleDelete(row)"
            >删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<style scoped>
.docs-page { max-width: 900px; margin: 0 auto; }

.stats-row { display: flex; gap: 14px; margin-bottom: 18px; }
.stat-card {
  flex: 1;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 20px;
  text-align: center;
}
.stat-value { font-size: 26px; font-weight: 700; color: var(--accent); }
.stat-label { font-size: 12px; color: var(--text-muted); margin-top: 4px; letter-spacing: 0.5px; }

.upload-section { margin-bottom: 0; }
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 12px;
  color: var(--text-primary);
}
.task-item {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 0; border-bottom: 1px solid var(--border);
}
.task-item:last-child { border-bottom: none; }
.task-filename { font-weight: 500; min-width: 180px; font-size: 13px; }
.task-detail { color: var(--success); font-size: 12px; }
.task-error {
  color: var(--danger); font-size: 12px;
  max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
</style>
