<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import { listDocuments, deleteDocument, uploadDocuments, getKbStats } from '../api'

const documents = ref([])
const stats = ref({ document_count: 0, chunk_count: 0, exists: false })
const uploading = ref(false)
const uploadProgress = ref(0)
const uploadFiles = ref([])

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

async function handleUpload() {
  const files = uploadFiles.value.map(item => item.raw).filter(Boolean)
  if (!files.length) {
    ElMessage.warning('Please select at least one file')
    return
  }
  uploading.value = true
  uploadProgress.value = 0
  try {
    const result = await uploadDocuments(files, (p) => {
      uploadProgress.value = p
    })
    if (result.success_count) {
      ElMessage.success(`批量上传完成：成功 ${result.success_count} 个，失败 ${result.failure_count} 个`)
    }
    if (result.failure_count) {
      const details = result.failures.map(item => `${item.filename}：${item.error}`).join('；')
      ElMessage.error(`上传失败 ${result.failure_count} 个：${details}`)
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
.docs-page {
  max-width: 900px;
  margin: 0 auto;
}

.stats-row {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
}
.stat-card {
  flex: 1;
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  text-align: center;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: var(--primary);
}
.stat-label {
  font-size: 13px;
  color: var(--text-muted);
  margin-top: 4px;
}

.upload-section {
  margin-bottom: 0;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  margin-bottom: 12px;
}
</style>
