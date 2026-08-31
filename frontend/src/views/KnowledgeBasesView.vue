<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { listKnowledgeBases, createKnowledgeBase, updateKnowledgeBase, deleteKnowledgeBase } from '../api'

const knowledgeBases = ref([])
const loading = ref(false)

// 新建/编辑对话框
const dialogVisible = ref(false)
const dialogMode = ref('create') // create | edit
const editingName = ref('') // 编辑时的原类别值（API 路径参数，改名后不变）
const form = ref({ name: '', display_name: '', description: '' })
const submitting = ref(false)

// 与后端一致：中文/字母/数字/下划线开头，仅含中文/字母/数字/下划线/连字符，1-32 位
const NAME_RULE = /^[\w\u4e00-\u9fff][\w\u4e00-\u9fff-]{0,31}$/

async function load() {
  loading.value = true
  try {
    knowledgeBases.value = await listKnowledgeBases()
  } catch (e) {
    ElMessage.error('加载知识库列表失败：' + e.message)
  } finally {
    loading.value = false
  }
}

function openCreate() {
  dialogMode.value = 'create'
  form.value = { name: '', display_name: '', description: '' }
  dialogVisible.value = true
}

function openEdit(row) {
  dialogMode.value = 'edit'
  editingName.value = row.name
  form.value = {
    name: row.name,
    display_name: row.display_name,
    description: row.description || ''
  }
  dialogVisible.value = true
}

async function handleSubmit() {
  const name = form.value.name.trim()
  // 创建与改名均需校验类别值
  if (!NAME_RULE.test(name)) {
    ElMessage.warning('类别值需以中文/字母/数字/下划线开头，仅含中文/字母/数字/下划线/连字符，长度 1-32')
    return
  }
  if (name.includes('"') || name.includes("'")) {
    ElMessage.warning('类别值不能包含引号')
    return
  }

  submitting.value = true
  try {
    if (dialogMode.value === 'create') {
      await createKnowledgeBase({
        name,
        display_name: form.value.display_name,
        description: form.value.description
      })
      ElMessage.success('创建成功')
    } else {
      await updateKnowledgeBase(editingName.value, {
        name: name !== editingName.value ? name : undefined,
        display_name: form.value.display_name,
        description: form.value.description
      })
      ElMessage.success('更新成功')
    }
    dialogVisible.value = false
    await load()
  } catch (e) {
    ElMessage.error((dialogMode.value === 'create' ? '创建' : '更新') + '失败：' + (e.response?.data?.detail || e.message))
  } finally {
    submitting.value = false
  }
}

async function handleDelete(row) {
  try {
    await ElMessageBox.confirm(
      `确认删除类别「${row.display_name}」？删除后不可恢复。`,
      '删除确认',
      { type: 'warning' }
    )
    await deleteKnowledgeBase(row.name)
    ElMessage.success('删除成功')
    await load()
  } catch (e) {
    if (e !== 'cancel') {
      ElMessage.error('删除失败：' + (e.response?.data?.detail || e.message))
    }
  }
}

onMounted(load)
</script>

<template>
  <div class="kb-page">
    <div class="page-header">
      <h2 class="page-title">知识库管理</h2>
      <el-button type="primary" @click="openCreate">新建类别</el-button>
    </div>

    <div class="card">
      <el-table v-loading="loading" :data="knowledgeBases" style="width: 100%" empty-text="暂无知识库类别">
        <el-table-column label="显示名" min-width="150">
          <template #default="{ row }">
            <span>{{ row.display_name }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="name" label="类别值" min-width="140" />
        <el-table-column prop="description" label="描述" min-width="180">
          <template #default="{ row }">{{ row.description || '—' }}</template>
        </el-table-column>
        <el-table-column prop="document_count" label="文档数" width="90" align="center" />
        <el-table-column prop="chunk_count" label="切块数" width="90" align="center" />
        <el-table-column label="创建时间" width="180">
          <template #default="{ row }">{{ row.created_at ? row.created_at.slice(0, 19).replace('T', ' ') : '—' }}</template>
        </el-table-column>
        <el-table-column label="操作" width="160" align="center">
          <template #default="{ row }">
            <el-button size="small" link type="primary" @click="openEdit(row)">编辑</el-button>
            <el-tooltip
              v-if="row.document_count > 0"
              content="请先删除该类别下所有文档"
              placement="top"
            >
              <span>
                <el-button size="small" link type="danger" disabled>删除</el-button>
              </span>
            </el-tooltip>
            <el-button
              v-else
              size="small"
              link
              type="danger"
              @click="handleDelete(row)"
            >删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 新建/编辑对话框 -->
    <el-dialog
      v-model="dialogVisible"
      :title="dialogMode === 'create' ? '新建知识库类别' : '编辑知识库类别'"
      width="480px"
      :close-on-click-modal="false"
    >
      <el-form label-width="80px">
        <el-form-item label="类别值" required>
          <el-input
            v-model="form.name"
            placeholder="如 法律法规（中文/字母/数字/下划线/连字符）"
          />
          <div class="form-tip">
            {{ dialogMode === 'create'
              ? '类别值写入文档的分类字段，用于检索过滤'
              : '修改类别值会把该类别下所有文档一并迁移到新类别' }}
          </div>
        </el-form-item>
        <el-form-item label="显示名">
          <el-input v-model="form.display_name" placeholder="如 法规库" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="form.description" type="textarea" :rows="3" placeholder="可选" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="handleSubmit">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.kb-page { max-width: 1000px; margin: 0 auto; }
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}
.page-title { margin: 0; }
.kb-tip {
  font-size: 13px;
  color: var(--text-muted);
  line-height: 1.6;
  margin-bottom: 14px;
  padding: 10px 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
}
.form-tip {
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.4;
  margin-top: 4px;
}
</style>
