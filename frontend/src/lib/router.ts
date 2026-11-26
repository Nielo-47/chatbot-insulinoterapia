export function navigateTo(path: string): void {
  window.location.hash = `#${path}`
}
