import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const streaming = ref(false)

  function addMessage(msg) {
    messages.value.push({ ...msg })
  }

  function updateLastAssistant(updater) {
    const last = messages.value[messages.value.length - 1]
    if (last && last.role === 'assistant') {
      // 创建新对象替换，确保 Vue 响应式触发
      const updated = { ...last }
      updater(updated)
      messages.value[messages.value.length - 1] = updated
    }
  }

  function clearMessages() {
    messages.value = []
  }

  function setStreaming(val) {
    streaming.value = val
  }

  return {
    messages,
    streaming,
    addMessage,
    updateLastAssistant,
    clearMessages,
    setStreaming
  }
})
