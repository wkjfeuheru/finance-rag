import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../store/auth'

const routes = [
  {
    path: '/login',
    name: 'login',
    component: () => import('../views/LoginView.vue'),
    meta: { title: '登录', public: true }
  },
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

// 全局前置守卫：未登录跳转 /login
router.beforeEach((to, from, next) => {
  const auth = useAuthStore()
  if (to.meta.public) {
    // 已登录访问登录页则跳转首页
    if (to.name === 'login' && auth.isAuthenticated()) {
      next({ name: 'chat' })
    } else {
      next()
    }
    return
  }
  if (!auth.isAuthenticated()) {
    next({ name: 'login' })
  } else {
    next()
  }
})

export default router
