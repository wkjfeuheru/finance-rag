import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/chat'
  },
  {
    path: '/chat',
    name: 'chat',
    component: () => import('../views/ChatView.vue'),
    meta: { title: '智能问答' }
  },
  {
    path: '/documents',
    name: 'documents',
    component: () => import('../views/DocumentsView.vue'),
    meta: { title: '文档管理' }
  },
  {
    path: '/evaluation',
    name: 'evaluation',
    component: () => import('../views/EvaluationView.vue'),
    meta: { title: '策略评估' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router

