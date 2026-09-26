<script setup>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import { listDocuments, deleteDocument, uploadDocuments, getTaskStatus, getKbStats, listKnowledgeBases, patchDocumentMetadata } from '../api'
import { categoryLabel } from '../constants/categories'

const documents = ref([])
const stats = ref({ document_count: 0, chunk_count: 0, exists: false })
const uploading = ref(false)
const uploadProgress = ref(0)
const uploadFiles = ref([])
const selectedCategory = ref('')     // 上传时选择的分类
const filterCategory = ref('')       // 列表分类筛选
const knowledgeBases = ref([])       // 知识库类别列表（内置四类 + 自定义）

// ---- 研报元数据人工修正 ----
// 抽取链路会抽错，没有修正入口这个维度就形同虚设
const metadataDialog = reactive({
  visible: false,
  saving: false,
  source: '',
  title: '',
  form: {
    security_code: '',
    security_name: '',
    industry_l1: '',
    industry_l2: '',
    report_type: '',
    broker: ''
  }
})

const REPORT_TYPES = ['个股', '行业', '宏观']

function openMetadataDialog(row) {
  metadataDialog.source = row.source
  metadataDialog.title = row.title || row.source
  metadataDialog.form = {
    security_code: row.security_code || '',
    security_name: row.security_name || '',
    industry_l1: row.industry_l1 || '',
    industry_l2: row.industry_l2 || '',
    report_type: row.report_type || '',
    broker: row.broker || ''
  }
  metadataDialog.visible = true
}

async function saveMetadata() {
  metadataDialog.saving = true
  try {
    await patchDocumentMetadata(metadataDialog.source, metadataDialog.form)
    ElMessage.success('元数据已更新（未重新嵌入，检索立即可用）')
    metadataDialog.visible = false
    await loadDocuments()
  } catch (e) {
    // 422 时的 detail 是中文说明（行业须为申万标准名等），直接展示
    const msg = e.response?.data?.detail || e.message
    ElMessage.error('更新失败：' + msg)
  } finally {
    metadataDialog.saving = false
  }
}

function isPlainDocument(row) {
  // 非研报文档（如内部制度）没有标的/行业，不显示「待确认」
  return !row.report_type && !row.security_code && !row.industry_l1
}

// 分类选项（类别值 -> 显示名），供上传选择与列表筛选使用
const categoryOptions = computed(() =>
  knowledgeBases.value.map(kb => ({ value: kb.name, label: kb.display_name }))
)

// 类别值 -> 显示名（表格分类标签用，动态支持自定义类别）
const categoryNameMap = computed(() =>
  Object.fromEntries(knowledgeBases.value.map(kb => [kb.name, kb.display_name]))
)

function displayCategory(value) {
  return categoryNameMap.value[value] || categoryLabel(value)
}

// 按分类筛选后的文档列表
const filteredDocuments = computed(() => {
  if (!filterCategory.value) return documents.value
  return documents.value.filter(d => (d.category || '') === filterCategory.value)
})

// 异步上传任务状态
const asyncTasks = ref([]) // [{task_id, filename, status, result, error}]
let _pollTimer = null

async function loadKnowledgeBases() {
  try {
    knowledgeBases.value = await listKnowledgeBases()
  } catch (e) {
    // 类别列表加载失败不阻断文档页，回退为内置四类
    knowledgeBases.value = []
  }
}

async function loadDocuments() {
  try {
    const [docs, st] = await Promise.all([
      listDocuments(),
      getKbStats()
    ])
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

async function handleUpload() {
  const files = uploadFiles.value.map(item => item.raw).filter(Boolean)
  if (!files.length) {
    ElMessage.warning('Please select at least one file')
    return
  }

  uploading.value = true
  uploadProgress.value = 0

  try {
    // 统一走异步上传，立即返回 task_id 列表
    const tasks = await uploadDocuments(files, (p) => {
      uploadProgress.value = p
    }, selectedCategory.value, '')

    for (const t of tasks) {
      asyncTasks.value.push({
        task_id: t.task_id,
        filename: t.filename,
        status: 'processing',
        result: null,
        error: null,
      })
    }
    ElMessage.info(`${tasks.length} 个文件已提交后台解析，请等待处理完成`)
    startPolling()

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
        task.finishedAt = Date.now()
        if (data.result?.skipped) {
          ElMessage.info(`文档已跳过（内容未变更）：${task.filename}`)
        } else {
          ElMessage.success(`文档处理完成：${task.filename}（${data.result?.chunk_count ?? 0} 个切块）`)
        }
        await loadDocuments()
      } else if (data.status === 'failed') {
        task.error = data.error
        task.finishedAt = Date.now()
        ElMessage.error(`文档处理失败：${task.filename}：${data.error}`)
      }
    } catch {
      // 轮询失败静默忽略
    }
  }
  // 已完成/失败任务保留 1 分钟后自动清理
  const now = Date.now()
  asyncTasks.value = asyncTasks.value.filter(
    t => t.status === 'processing' || now - (t.finishedAt || 0) < 60000
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
    await deleteDocument(doc.source, '')
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
  loadKnowledgeBases()
  loadDocuments()
})
</script>

<template>
  <div class="docs-page">
    <div class="page-header">
      <h2 class="page-title">文档管理</h2>
    </div>

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
      <div class="upload-category">
        <span class="upload-category-label">文档分类：</span>
        <el-select
          v-model="selectedCategory"
          clearable
          placeholder="未分类"
          style="width: 200px"
        >
          <el-option
            v-for="c in categoryOptions"
            :key="c.value"
            :label="c.label"
            :value="c.value"
          />
        </el-select>
      </div>
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
          {{ task.result.skipped ? '已跳过（内容未变更）' : `${task.result.chunk_count} 个切块` }}
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
        <div class="list-actions">
          <el-select
            v-model="filterCategory"
            clearable
            placeholder="全部类别"
            style="width: 160px; margin-right: 8px"
          >
            <el-option
              v-for="c in categoryOptions"
              :key="c.value"
              :label="c.label"
              :value="c.value"
            />
          </el-select>
          <el-button size="small" @click="loadDocuments">刷新</el-button>
        </div>
      </div>
      <el-table :data="filteredDocuments" style="width: 100%" empty-text="暂无文档，请上传">
        <el-table-column prop="title" label="文档标题" min-width="180" />
        <el-table-column label="文件名" min-width="180">
          <template #default="{ row }">{{ formatSource(row.source) }}</template>
        </el-table-column>
        <el-table-column label="分类" width="140" align="center">
          <template #default="{ row }">
            <el-tag size="small">{{ displayCategory(row.category) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="研报元数据" min-width="200">
          <template #default="{ row }">
            <div class="meta-cell">
              <el-tag v-if="row.security_code" size="small" type="info">{{ row.security_code }}</el-tag>
              <el-tag v-if="row.industry_l1" size="small" type="info">{{ row.industry_l1 }}</el-tag>
              <el-tag v-if="row.report_type" size="small" type="info">{{ row.report_type }}</el-tag>
              <el-tag v-if="row.needs_review" size="small" type="warning">待确认</el-tag>
              <span v-if="isPlainDocument(row)" class="meta-empty">—</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="chunk_count" label="向量块数" width="110" align="center" />
        <el-table-column label="操作" width="150" align="center">
          <template #default="{ row }">
            <el-button
              type="primary"
              size="small"
              link
              @click="openMetadataDialog(row)"
            >元数据</el-button>
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

    <!-- 研报元数据人工修正 -->
    <el-dialog v-model="metadataDialog.visible" :title="`修正元数据 — ${metadataDialog.title}`" width="560px">
      <el-form label-width="96px">
        <el-form-item label="证券代码">
          <el-input v-model="metadataDialog.form.security_code" placeholder="6 位 A 股代码，如 600519" />
        </el-form-item>
        <el-form-item label="证券简称">
          <el-input v-model="metadataDialog.form.security_name" placeholder="如 贵州茅台" />
        </el-form-item>
        <el-form-item label="申万一级">
          <el-input v-model="metadataDialog.form.industry_l1" placeholder="必须用申万标准名，如 食品饮料" />
        </el-form-item>
        <el-form-item label="申万二级">
          <el-input v-model="metadataDialog.form.industry_l2" placeholder="必须属于所选一级，如 白酒Ⅱ" />
        </el-form-item>
        <el-form-item label="报告类型">
          <el-select v-model="metadataDialog.form.report_type" clearable placeholder="未设置">
            <el-option v-for="t in REPORT_TYPES" :key="t" :label="t" :value="t" />
          </el-select>
        </el-form-item>
        <el-form-item label="发布机构">
          <el-input v-model="metadataDialog.form.broker" placeholder="如 中信证券" />
        </el-form-item>
      </el-form>
      <div class="meta-hint">
        行业名必须是申万标准名（如「白酒Ⅱ」而不是「白酒」），否则该文档无法被行业维度检索到。
        留空表示清空该字段。
      </div>
      <template #footer>
        <el-button @click="metadataDialog.visible = false">取消</el-button>
        <el-button type="primary" :loading="metadataDialog.saving" @click="saveMetadata">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.docs-page { max-width: 900px; margin: 0 auto; }

.meta-cell {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  align-items: center;
}
.meta-empty { color: var(--text-muted); }
.meta-hint {
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.6;
  margin-top: -4px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 18px;
}
.page-title { margin: 0; }

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
.upload-category {
  display: flex;
  align-items: center;
  margin-top: 12px;
}
.upload-category-label {
  font-size: 13px;
  color: var(--text-secondary);
  margin-right: 8px;
}
.list-actions {
  display: flex;
  align-items: center;
}
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
