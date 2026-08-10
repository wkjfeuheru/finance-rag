<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '../store/auth'

const router = useRouter()
const authStore = useAuthStore()

const form = reactive({ username: '', password: '' })
const loading = ref(false)
const formRef = ref(null)

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

async function handleLogin() {
  if (!formRef.value) return
  try { await formRef.value.validate() } catch { return }
  loading.value = true
  try {
    await authStore.login(form.username, form.password)
    ElMessage.success('登录成功')
    router.push('/chat')
  } catch (err) {
    const msg = err.response?.data?.detail || '登录失败，请检查用户名和密码'
    ElMessage.error(msg)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <div class="login-bg"></div>
    <div class="login-card">
      <div class="login-header">
        <div class="login-icon">R</div>
        <h2>金融 RAG 知识平台</h2>
        <p>智能检索 · 精准问答</p>
      </div>
      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-position="top"
        @keyup.enter="handleLogin"
      >
        <el-form-item label="用户名" prop="username">
          <el-input v-model="form.username" placeholder="请输入用户名" autocomplete="username" />
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码"
            show-password
            autocomplete="current-password"
          />
        </el-form-item>
        <el-button type="primary" class="login-btn" :loading="loading" @click="handleLogin">
          登录
        </el-button>
      </el-form>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-primary);
  position: relative;
  overflow: hidden;
}

.login-bg {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 30% 20%, rgba(201, 168, 76, 0.04) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 70% 80%, rgba(201, 168, 76, 0.03) 0%, transparent 50%);
  pointer-events: none;
}

.login-card {
  width: 380px;
  padding: 44px 36px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  position: relative;
  z-index: 1;
}

.login-header { text-align: center; margin-bottom: 32px; }

.login-icon {
  width: 48px;
  height: 48px;
  margin: 0 auto 14px;
  background: linear-gradient(135deg, var(--accent), #8b6914);
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  font-weight: 700;
  color: #0b1019;
  font-family: var(--font-display);
}

.login-header h2 {
  margin: 0 0 6px;
  font-size: 20px;
  color: var(--text-primary);
  font-family: var(--font-display);
  font-weight: 600;
}

.login-header p {
  margin: 0;
  color: var(--text-muted);
  font-size: 13px;
  letter-spacing: 1px;
}

.login-btn {
  width: 100%;
  margin-top: 6px;
  height: 40px;
  font-size: 14px;
}
</style>
