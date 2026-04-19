// Shared types mirroring app/schemas.py.
// Copied by hand — keep in sync with the backend.

export type JobStatus = 'queued' | 'running' | 'done' | 'failed'

export type PIIType =
  | 'PASSPORT'
  | 'INN'
  | 'SNILS'
  | 'PHONE'
  | 'EMAIL'
  | 'ADDRESS'
  | 'PERSON'

export type PIISource = 'regex' | 'natasha' | 'llm' | 'word_phone' | string

export interface JobCreated {
  job_id: string
  status: JobStatus
  filename: string | null
}

export interface JobInfo {
  id: string
  created_at: string
  status: JobStatus
  input_filename: string | null
  duration_sec: number | null
  error: string | null
  pii_count: number | null
}

export interface Word {
  word: string
  start: number
  end: number
  speaker: string | null
  probability: number
  /** Present on PII-tagged words. */
  pii_type?: PIIType | null
  /** Only present in the redacted transcript. */
  original_word?: string | null
}

export interface Transcript {
  language: string
  duration: number
  speakers?: unknown
  words: Word[]
  text: string
  pii_count: number
}

export interface EventItem {
  id: number
  timestamp: string
  pii_type: string
  text: string
  start_sec: number | null
  end_sec: number | null
  source: string | null
  confidence: number | null
}

export interface EventsResponse {
  job_id: string
  count: number
  events: EventItem[]
}

export type TranscriptVersion = 'full' | 'redacted'
export type AudioVersion = 'original' | 'redacted'
