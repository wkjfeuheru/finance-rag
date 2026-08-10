import { defineStore } from 'pinia'
import { ref } from 'vue'
import { login as loginApi, getToken, clearToken } from '../api'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(getToken() || '')
  const username = ref('')

  function isAuthenticated() {
    return !!token.value
  }

  async function login(usernameInput, password) {
    const data = await loginApi(usernameInput, password)
    token.value = data.access_token
    username.value = usernameInput
  }

  function logout() {
    clearToken()
    token.value = ''
    username.value = ''
  }

  return { token, username, isAuthenticated, login, logout }
})
