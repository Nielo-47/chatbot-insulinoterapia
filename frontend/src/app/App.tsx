import { useEffect, useState } from 'react'

import { ApiError, deleteAccount, getCurrentUser, checkHealth } from '../lib/api'
import { getAuthentikLogoutUrl } from '../lib/env'
import { ChatPage } from '../features/chat/ChatPage'
import { SignInPage } from '../features/auth/SignInPage'
import type { AuthStatus, BackendStatus } from '../types/app'

type CurrentUser = {
  id: number
  username: string
}

function App() {
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null)
  const [isBootstrapping, setIsBootstrapping] = useState(true)
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking')
  const [authStatus, setAuthStatus] = useState<AuthStatus>('checking')

  useEffect(() => {
    void (async () => {
      let backendOnline = false

      try {
        await checkHealth()
        backendOnline = true
        setBackendStatus('online')
      } catch {
        setBackendStatus('offline')
      }

      if (!backendOnline) {
        setAuthStatus('unknown')
        setIsBootstrapping(false)
        return
      }

      // The Authentik session lives in browser cookies and is enforced by the
      // nginx forward-auth proxy: /auth/me only succeeds when a valid session
      // exists, so probing it reveals the auth state (no client-side token).
      try {
        const user = await getCurrentUser()
        setCurrentUser(user)
        setAuthStatus('authenticated')
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          setAuthStatus('signed_out')
        } else {
          setAuthStatus('unknown')
        }
      } finally {
        setIsBootstrapping(false)
      }
    })()
  }, [])

  const handleLogout = async (reason: 'manual' | 'expired' | 'deleted' = 'manual') => {
    if (reason === 'manual' || reason === 'deleted') {
      // Sign out of Authentik so the session cookie is destroyed, then the
      // browser lands back here unauthenticated.
      window.location.assign(getAuthentikLogoutUrl())
      return
    }
    setCurrentUser(null)
    setAuthStatus('expired')
  }

  const handleDeleteAccount = async () => {
    await deleteAccount()
    await handleLogout('deleted')
  }

  if (isBootstrapping) {
    return <div className="min-h-screen bg-slate-950" />
  }

  if (!currentUser) {
    return <SignInPage backendStatus={backendStatus} authStatus={authStatus} />
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
