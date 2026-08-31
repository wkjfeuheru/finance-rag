<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from './store/auth'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const navItems = [
  { path: '/chat', label: '智能问答', icon: '💬' },
  { path: '/compliance', label: '合规审查', icon: '⚖️' },
  { path: '/documents', label: '文档管理', icon: '📄' },
  { path: '/knowledge-bases', label: '知识库管理', icon: '📚' }
]

const activePath = computed(() => route.path)
const isLoginPage = computed(() => route.name === 'login')

function navigate(path) {
  router.push(path)
}

function handleLogout() {
  authStore.logout()
  router.push('/login')
}
</script>

<template>
  <router-view v-if="isLoginPage" />
  <div v-else class="app-layout">
    <aside class="app-sidebar">
      <div class="app-logo">
        <div class="app-logo-icon">R</div>
        <span class="app-logo-text">金融 RAG 平台</span>
      </div>
      <nav class="app-nav">
        <div
          v-for="item in navItems"
          :key="item.path"
          class="nav-item"
          :class="{ active: activePath === item.path }"
          @click="navigate(item.path)"
        >
          <span class="nav-item-icon">{{ item.icon }}</span>
          <span class="nav-item-label">{{ item.label }}</span>
        </div>
      </nav>
      <div class="app-footer">
        <button class="logout-btn" @click="handleLogout">退出登录</button>
      </div>
    </aside>
    <main class="app-main">
      <div class="app-content">
        <router-view />
      </div>
    </main>
  </div>
</template>
