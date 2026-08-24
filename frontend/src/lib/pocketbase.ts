import PocketBase from 'pocketbase'

import { env } from './env'

// Single shared client for the whole app. The auth token + record are
// persisted to localStorage by pocketbase-js, so a page reload keeps the user
// signed in. All traffic goes through the same-origin /pb nginx proxy, so no
// third-party origin is ever contacted directly.
export const pocketbase = new PocketBase(env.pocketbaseUrl)

// The backend rejects expired tokens with 401; the token lifetime is short
// (1h), so auto-cancellation of pending requests is left disabled and the api
// client handles 401 as "sign out".
pocketbase.autoCancellation(false)
