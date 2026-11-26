import { useEffect, useState } from 'react'

import { ApiError, deleteAccount, getCurrentUser, checkHealth } from '../lib/api'
import { pocketbase } from '../lib/pocketbase'
import { ChatPage } from '../features/chat/ChatPage'
import { SignInPage } from '../features/auth/SignInPage'
import { SignUpPage } from '../features/auth/SignUpPage'
import type { AuthStatus, BackendStatus } from '../types/app'

type CurrentUser = {
  id: string
  username: string
}

function readRoute(): string {
  // Hash-based routing: no hash -> login, `#/signin` -> registration.
  return window.location.hash.startsWith('#/signin') ? '/signin' : '/'
}

function App() {
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null)
  const [route, setRoute] = useState<string>(() => readRoute())
  const [isBootstrapping, setIsBootstrapping] = useState(true)
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking')
  const [authStatus, setAuthStatus] = useState<AuthStatus>('checking')

  useEffect(() => {
    const handleHashChange = () => setRoute(readRoute())
    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

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

      // The PocketBase session lives in localStorage; /auth/me only succeeds
      // when the stored token is still valid, so probing it reveals the auth
      // state.
      if (!pocketbase.authStore.isValid) {
        setAuthStatus('signed_out')
        setIsBootstrapping(false)
        return
      }

      try {
        const user = await getCurrentUser()
        setCurrentUser(user)
        setAuthStatus('authenticated')
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          pocketbase.authStore.clear()
          setAuthStatus('signed_out')
        } else {
          setAuthStatus('unknown')
        }
      } finally {
        setIsBootstrapping(false)
      }
    })()
  }, [])

  useEffect(() => {
    // Fires on sign-in (authWithPassword), sign-out and token changes.
    const unsubscribe = pocketbase.authStore.onChange(() => {
      if (pocketbase.authStore.isValid) {
        void (async () => {
          try {
            const user = await getCurrentUser()
            setCurrentUser(user)
            setAuthStatus('authenticated')
          } catch {
            // The bootstrap flow re-evaluates on next reload; ignore transient errors.
          }
        })()
      } else {
        setCurrentUser(null)
        setAuthStatus('signed_out')
      }
    })
    return () => unsubscribe()
  }, [])

  const handleLogout = async (reason: 'manual' | 'expired' | 'deleted' = 'manual') => {
    pocketbase.authStore.clear()
    setCurrentUser(null)
    setAuthStatus(reason === 'expired' ? 'expired' : 'signed_out')
  }

  const handleDeleteAccount = async () => {
    await deleteAccount()
    await handleLogout('deleted')
  }

  if (isBootstrapping) {
    return <div className="min-h-screen bg-slate-950" />
  }

  if (!currentUser) {
    return route === '/signin' ? (
      <SignUpPage backendStatus={backendStatus} authStatus={authStatus} />
    ) : (
      <SignInPage backendStatus={backendStatus} authStatus={authStatus} />
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
