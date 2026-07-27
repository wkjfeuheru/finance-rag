<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()

const navItems = [
  { path: '/chat', label: '智能问答', icon: '💬' },
  { path: '/documents', label: '文档管理', icon: '📄' },
  { path: '/evaluation', label: '策略评估', icon: '📊' }
]

const activePath = computed(() => route.path)

function navigate(path) {
  router.push(path)
}
</script>

<template>
  <div class="app-layout">
    <aside class="app-sidebar">
      <div class="app-logo">
        <span>💰</span>
        <span>金融 RAG 平台</span>
      </div>
      <nav class="app-nav">
        <div
          v-for="item in navItems"
          :key="item.path"
          class="nav-item"
          :class="{ active: activePath === item.path }"
          @click="navigate(item.path)"
        >
          <span>{{ item.icon }}</span>
          <span>{{ item.label }}</span>
        </div>
      </nav>
    </aside>
    <main class="app-main">
      <div class="app-content">
        <router-view />
      </div>
    </main>
  </div>
</template>
