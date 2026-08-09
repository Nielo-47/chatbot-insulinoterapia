export function truncate(text: string, maxLength = 200): string {
  if (text.length <= maxLength) {
    return text
  }
  return text.slice(0, maxLength).trimEnd() + '…'
}

export function friendlyFileName(path: string): string {
  return path.split('/').pop()?.replace(/\.[^.]+$/, '') || path
}
