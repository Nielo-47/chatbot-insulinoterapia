import { useEffect, useState } from 'react'

import { ApiError, deleteAccount, getCurrentUser, checkHealth } from '../lib/api'
import { supabase } from '../lib/supabase'
import { ChatPage } from '../features/chat/ChatPage'
import { SignInPage } from '../features/auth/SignInPage'
import type { AuthStatus, BackendStatus } from '../types/app'

type CurrentUser = {
  id: string
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

      // The Supabase session lives in localStorage; /auth/me only succeeds when
      // the session yields a valid access token, so probing it reveals the
      // auth state.
      const {
        data: { session },
      } = await supabase.auth.getSession()

      if (!session) {
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
          await supabase.auth.signOut()
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
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event) => {
      if (event === 'SIGNED_IN') {
        void (async () => {
          try {
            const user = await getCurrentUser()
            setCurrentUser(user)
            setAuthStatus('authenticated')
          } catch {
            // The bootstrap flow re-evaluates on next reload; ignore transient errors.
          }
        })()
      } else if (event === 'SIGNED_OUT') {
        setCurrentUser(null)
        setAuthStatus('signed_out')
      }
    })
    return () => subscription.unsubscribe()
  }, [])

  const handleLogout = async (reason: 'manual' | 'expired' | 'deleted' = 'manual') => {
    await supabase.auth.signOut()
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
