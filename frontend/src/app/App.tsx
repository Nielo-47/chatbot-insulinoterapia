import { useEffect, useState } from 'react'

import { clearAuthSession, getCurrentUser, login as loginRequest } from '../lib/api'
import { authStorage } from '../lib/auth'
import { ChatPage } from '../features/chat/ChatPage'
import { LoginPage } from '../features/auth/LoginPage'

type CurrentUser = {
  id: number
  username: string
}

function App() {
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null)
  const [isBootstrapping, setIsBootstrapping] = useState(true)
  const [authError, setAuthError] = useState<string | null>(null)
  const [isLoggingIn, setIsLoggingIn] = useState(false)

  useEffect(() => {
    void (async () => {
      const token = authStorage.getToken()

      if (!token) {
        setIsBootstrapping(false)
        return
      }

      try {
        const user = await getCurrentUser()
        setCurrentUser(user)
      } catch {
        await clearAuthSession()
      } finally {
        setIsBootstrapping(false)
      }
    })()
  }, [])

  const handleLogin = async (username: string, password: string) => {
    setIsLoggingIn(true)
    setAuthError(null)

    try {
      await loginRequest(username, password)
      const user = await getCurrentUser()
      setCurrentUser(user)
    } catch (error) {
      await clearAuthSession()
      setAuthError(error instanceof Error ? error.message : 'Nao foi possivel entrar.')
    } finally {
      setIsLoggingIn(false)
    }
  }

  const handleLogout = async () => {
    await clearAuthSession()
    setCurrentUser(null)
  }

  if (isBootstrapping) {
    return <div className="min-h-screen bg-slate-950" />
  }

  if (!currentUser) {
    return <LoginPage onLogin={handleLogin} errorMessage={authError} isSubmitting={isLoggingIn} />
  }

  return <ChatPage username={currentUser.username} onLogout={handleLogout} />
}

export default App
