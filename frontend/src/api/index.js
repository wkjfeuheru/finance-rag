import axios from 'axios'

const TOKEN_KEY = 'finance_rag_token'

export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t)
export const clearToken = () => localStorage.removeItem(TOKEN_KEY)

const http = axios.create({
  baseURL: '/api',
  timeout: 60000
})

// 请求拦截器：自动附加 Authorization header
http.interceptors.request.use(config => {
  const token = getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// uvicorn 热重载时，幂等 GET 请求对临时网络错误或 5xx 自动重试。
http.interceptors.response.use(
  response => response,
  async error => {
    const config = error.config
    const status = error.response?.status

    // 401：清 token 并跳登录页
    if (status === 401) {
      clearToken()
      if (window.location.pathname !== '/login') {
        window.location.href = '/login'
      }
      return Promise.reject(error)
    }

    const retryable = config?.method?.toLowerCase() === 'get' &&
      (!error.response || (status >= 500 && status < 600))
    config.__retryCount = config.__retryCount || 0
    if (retryable && config.__retryCount < 2) {
      config.__retryCount += 1
      await new Promise(resolve => setTimeout(resolve, config.__retryCount * 600))
      return http(config)
    }
    return Promise.reject(error)
  }
)

// ---------------------------------------------------------------------------
// 健康检查
// ---------------------------------------------------------------------------

export const getHealth = () => http.get('/health').then(r => r.data)

// ---------------------------------------------------------------------------
// 鉴权
// ---------------------------------------------------------------------------

export const login = (username, password) =>
  http.post('/auth/login', { username, password }).then(r => {
    setToken(r.data.access_token)
    return r.data
  })

// ---------------------------------------------------------------------------
// 策略评估（ragas）
// ---------------------------------------------------------------------------

export const getTestQueries = () => http.get('/test-queries').then(r => r.data)
export const evaluateStrategy = (payload) => http.post('/evaluate-strategy', payload, { timeout: 600000 }).then(r => r.data)

// ---------------------------------------------------------------------------
// 文档管理
// ---------------------------------------------------------------------------

export const listDocuments = () => http.get('/documents').then(r => r.data)
export const deleteDocument = (source) => http.delete(`/documents/${encodeURIComponent(source)}`).then(r => r.data)
export const uploadDocuments = (files, onProgress) => {
  const form = new FormData()
  files.forEach(file => form.append('files', file))
  return http.post('/documents/upload', form, {
    timeout: 300000,
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    }
  }).then(r => r.data)
}

export const uploadDocumentsAsync = (files, onProgress) => {
  const form = new FormData()
  files.forEach(file => form.append('files', file))
  return http.post('/documents/upload-async', form, {
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    }
  }).then(r => r.data)
}

export const getTaskStatus = (taskId) => http.get(`/tasks/${taskId}`).then(r => r.data)

export const getKbStats = () => http.get('/kb/stats').then(r => r.data)

// ---------------------------------------------------------------------------
// 非流式问答
// ---------------------------------------------------------------------------

export const chat = (payload) => http.post('/chat', payload).then(r => r.data)

// ---------------------------------------------------------------------------
// 流式问答（SSE）
// 接收回调 onEvent(event)，返回 AbortController 用于中止
// ---------------------------------------------------------------------------

export function chatStream(payload, onEvent, onError) {
  const controller = new AbortController()

  fetch('/api/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(getToken() ? { Authorization: `Bearer ${getToken()}` } : {})
    },
    body: JSON.stringify(payload),
    signal: controller.signal
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // 解析 SSE 事件（以空行分隔，兼容 \n\n 和 \r\n\r\n）
        const blocks = buffer.split(/\r?\n\r?\n/)
        buffer = blocks.pop() || ''

        for (const block of blocks) {
          const event = parseSSEBlock(block)
          if (event) onEvent(event)
        }
      }
      // 处理最后残余
      if (buffer.trim()) {
        const event = parseSSEBlock(buffer)
        if (event) onEvent(event)
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError' && onError) onError(err)
    })

  return controller
}

function parseSSEBlock(block) {
  let eventType = 'message'
  let dataStr = ''
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      // 保留 data 内容中的空格，只去除 "data:" 前缀后的一个可选空格
      const raw = line.slice(5)
      dataStr += raw.startsWith(' ') ? raw.slice(1) : raw
    }
  }
  if (!dataStr) return null
  try {
    const data = JSON.parse(dataStr)
    return { type: eventType, ...data }
  } catch {
    return { type: eventType, data: dataStr }
  }
}


