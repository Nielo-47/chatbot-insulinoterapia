import { createClient } from '@supabase/supabase-js'

import { env } from './env'

// Single shared client for the whole app. The session (access + refresh
// tokens) is persisted to localStorage by supabase-js, so a page reload keeps
// the user signed in. PKCE is used so a refresh token is never exposed to
// third parties (the app is served over a stable https domain via ngrok).
export const supabase = createClient(env.supabaseUrl, env.supabaseAnonKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
    flowType: 'pkce',
  },
})
