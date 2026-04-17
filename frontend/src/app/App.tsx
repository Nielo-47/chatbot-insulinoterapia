import { useEffect, useState } from 'react'

import { ApiError, clearAuthSession, deleteAccount, getCurrentUser, login as loginRequest, checkHealth } from '../lib/api'
import { authStorage } from '../lib/auth'
import { ChatPage } from '../features/chat/ChatPage'
import { LoginPage } from '../features/auth/LoginPage'
import type { AuthStatus, BackendStatus } from '../types/app'

type CurrentUser = {
  id: number
  username: string
}

function App() {
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null)
  const [isBootstrapping, setIsBootstrapping] = useState(true)
  const [authError, setAuthError] = useState<string | null>(null)
  const [isLoggingIn, setIsLoggingIn] = useState(false)
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking')
  const [authStatus, setAuthStatus] = useState<AuthStatus>('checking')

  useEffect(() => {
    void (async () => {
      const token = authStorage.getToken()
      let backendOnline = false

      try {
        await checkHealth()
        backendOnline = true
        setBackendStatus('online')
      } catch {
        setBackendStatus('offline')
      }

      if (!token) {
        setAuthStatus('signed_out')
        setIsBootstrapping(false)
        return
      }

      try {
        if (backendOnline) {
          const user = await getCurrentUser()
          setCurrentUser(user)
          setAuthStatus('authenticated')
        } else {
          setAuthStatus('unknown')
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          setAuthStatus('invalid')
        } else {
          setAuthStatus('unknown')
        }
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
      setAuthStatus('authenticated')
    } catch (error) {
      await clearAuthSession()
      if (error instanceof ApiError && error.status === 401) {
        setAuthStatus('invalid')
      } else if (error instanceof ApiError && error.status >= 500) {
        setAuthStatus('unknown')
      } else {
        setAuthStatus('signed_out')
      }
      setAuthError(error instanceof Error ? error.message : 'Nao foi possivel entrar.')
    } finally {
      setIsLoggingIn(false)
    }
  }

  const handleLogout = async (reason: 'manual' | 'expired' | 'deleted' = 'manual') => {
    await clearAuthSession()
    setCurrentUser(null)
    setAuthStatus(reason === 'expired' ? 'invalid' : 'signed_out')
  }

  const handleDeleteAccount = async () => {
    await deleteAccount()
    await handleLogout('deleted')
  }

  if (isBootstrapping) {
    return <div className="min-h-screen bg-slate-950" />
  }

  if (!currentUser) {
    return (
      <LoginPage
        onLogin={handleLogin}
        errorMessage={authError}
        isSubmitting={isLoggingIn}
        backendStatus={backendStatus}
        authStatus={authStatus}
      />
    )
  }

  return (
    <ChatPage
      username={currentUser.username}
      backendStatus={backendStatus}
      authStatus={authStatus}
      onLogout={handleLogout}
      onDeleteAccount={handleDeleteAccount}
    />
  )
}

export default App
