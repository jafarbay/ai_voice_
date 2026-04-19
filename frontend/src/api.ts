// Thin typed wrapper around the backend REST API.
// Uses fetch + relative URLs so the Vite dev proxy handles host/CORS.

import type {
  AudioVersion,
  EventsResponse,
  JobCreated,
  JobInfo,
  Transcript,
  TranscriptVersion,
} from './types'

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(`HTTP ${status}: ${detail}`)
    this.status = status
    this.detail = detail
  }
}

async function parseError(res: Response): Promise<ApiError> {
  let detail = res.statusText || 'Request failed'
  try {
    const body = await res.json()
    if (body && typeof body.detail === 'string') {
      detail = body.detail
    }
  } catch {
    // non-JSON body; stick with statusText
  }
  return new ApiError(res.status, detail)
}

export async function uploadJob(file: File): Promise<JobCreated> {
  const form = new FormData()
  form.append('file', file, file.name)
  const res = await fetch('/jobs', { method: 'POST', body: form })
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function listJobs(): Promise<JobInfo[]> {
  const res = await fetch('/jobs')
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function getJob(id: string): Promise<JobInfo> {
  const res = await fetch(`/jobs/${encodeURIComponent(id)}`)
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function getTranscript(
  id: string,
  version: TranscriptVersion,
): Promise<Transcript> {
  const res = await fetch(
    `/jobs/${encodeURIComponent(id)}/transcript?version=${version}`,
  )
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export function audioUrl(id: string, version: AudioVersion): string {
  return `/jobs/${encodeURIComponent(id)}/audio?version=${version}`
}

export async function getEvents(id: string): Promise<EventsResponse> {
  const res = await fetch(`/jobs/${encodeURIComponent(id)}/events`)
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export function exportUrl(id: string): string {
  return `/jobs/${encodeURIComponent(id)}/export`
}
