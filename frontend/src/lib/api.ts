export type ApiStatus = { status: string; version?: string; detail?: string }
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
export async function getHealth(): Promise<ApiStatus> {
  const response = await fetch(`${API_BASE}/health/`)
  if (!response.ok) throw new Error('Health endpoint unavailable')
  return response.json()
}
export { API_BASE }
