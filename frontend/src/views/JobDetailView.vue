<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  ApiError,
  audioUrl,
  exportUrl,
  getEvents,
  getJob,
  getTranscript,
} from '../api'
import type {
  EventItem,
  JobInfo,
  Transcript,
  TranscriptVersion,
} from '../types'
import StatusBadge from '../components/StatusBadge.vue'
import TranscriptWords from '../components/TranscriptWords.vue'

const props = defineProps<{ id: string }>()

const POLL_INTERVAL_MS = 2000

const job = ref<JobInfo | null>(null)
const transcriptFull = ref<Transcript | null>(null)
const transcriptRedacted = ref<Transcript | null>(null)
const events = ref<EventItem[]>([])
const error = ref<string | null>(null)
const loading = ref(false)

const activeTab = ref<TranscriptVersion>('full')

const redactedAudioRef = ref<HTMLAudioElement | null>(null)

const originalAudioSrc = computed(() => audioUrl(props.id, 'original'))
const redactedAudioSrc = computed(() => audioUrl(props.id, 'redacted'))

let pollTimer: number | null = null

function stopPolling() {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

function startPolling() {
  stopPolling()
  pollTimer = window.setInterval(() => {
    void refreshJob()
  }, POLL_INTERVAL_MS)
}

async function refreshJob() {
  try {
    const info = await getJob(props.id)
    job.value = info
    if (info.status === 'done' || info.status === 'failed') {
      stopPolling()
      if (info.status === 'done') {
        await loadArtifacts()
      }
    }
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      error.value = `Задача '${props.id}' не найдена`
      stopPolling()
    } else {
      // Transient errors during polling are noted but don't stop the poller.
      error.value =
        e instanceof ApiError ? e.detail : (e as Error)?.message ?? 'Ошибка'
    }
  }
}

async function loadArtifacts() {
  // Only fetch once per load.
  const tasks: Promise<unknown>[] = []
  if (!transcriptFull.value) {
    tasks.push(
      getTranscript(props.id, 'full').then((t) => (transcriptFull.value = t)),
    )
  }
  if (!transcriptRedacted.value) {
    tasks.push(
      getTranscript(props.id, 'redacted').then(
        (t) => (transcriptRedacted.value = t),
      ),
    )
  }
  if (events.value.length === 0) {
    tasks.push(getEvents(props.id).then((e) => (events.value = e.events)))
  }
  try {
    await Promise.all(tasks)
  } catch (e) {
    error.value =
      e instanceof ApiError
        ? e.detail
        : (e as Error)?.message ?? 'Не удалось загрузить артефакты'
  }
}

async function init() {
  loading.value = true
  error.value = null
  transcriptFull.value = null
  transcriptRedacted.value = null
  events.value = []
  try {
    const info = await getJob(props.id)
    job.value = info
    if (info.status === 'done') {
      await loadArtifacts()
    } else if (info.status === 'queued' || info.status === 'running') {
      startPolling()
    }
  } catch (e) {
    error.value =
      e instanceof ApiError ? e.detail : (e as Error)?.message ?? 'Ошибка'
  } finally {
    loading.value = false
  }
}

function seekRedacted(ev: EventItem) {
  if (ev.start_sec == null) return
  const audio = redactedAudioRef.value
  if (!audio) return
  audio.currentTime = ev.start_sec
  void audio.play().catch(() => {
    /* autoplay may be blocked; user can press play */
  })
}

function formatTime(ts: string): string {
  const d = new Date(ts)
  if (isNaN(d.getTime())) return ts
  return d.toLocaleString('ru-RU')
}

function formatSeconds(s: number | null | undefined): string {
  if (s == null) return '—'
  return s.toFixed(2) + ' с'
}

function formatConfidence(c: number | null | undefined): string {
  if (c == null) return '—'
  return (c * 100).toFixed(0) + '%'
}

const activeTranscript = computed(() =>
  activeTab.value === 'full' ? transcriptFull.value : transcriptRedacted.value,
)

onMounted(init)
onBeforeUnmount(stopPolling)

// Route can change in place (e.g. navigating from one detail to another).
watch(
  () => props.id,
  () => {
    stopPolling()
    void init()
  },
)
</script>

<template>
  <section class="card">
    <div class="header-row">
      <div>
        <h2 style="margin-bottom: 4px">
          Задача <span class="mono">{{ id }}</span>
        </h2>
        <div v-if="job" class="muted" style="font-size: 13px">
          Создана {{ formatTime(job.created_at) }}
        </div>
      </div>
      <StatusBadge v-if="job" :status="job.status" />
    </div>

    <div v-if="error" class="error-box">{{ error }}</div>
    <p v-if="loading && !job" class="muted">Загрузка...</p>

    <dl v-if="job" class="meta">
      <dt>Файл</dt>
      <dd>{{ job.input_filename ?? '—' }}</dd>
      <dt>Длительность</dt>
      <dd>{{ job.duration_sec != null ? job.duration_sec.toFixed(2) + ' с' : '—' }}</dd>
      <dt>Найдено PII</dt>
      <dd>{{ job.pii_count ?? '—' }}</dd>
      <dt v-if="job.error">Ошибка</dt>
      <dd v-if="job.error" style="color: #991b1b">{{ job.error }}</dd>
    </dl>

    <p
      v-if="job && (job.status === 'queued' || job.status === 'running')"
      class="muted"
      style="margin-top: 12px"
    >
      Обработка идёт, страница обновляется автоматически каждые 2 с.
    </p>
  </section>

  <section v-if="job?.status === 'done'" class="card">
    <h2>Аудио</h2>
    <div class="row">
      <div>
        <div class="muted" style="font-size: 13px">Исходное</div>
        <audio controls preload="metadata" :src="originalAudioSrc"></audio>
      </div>
      <div>
        <div class="muted" style="font-size: 13px">Обезличенное</div>
        <audio
          ref="redactedAudioRef"
          controls
          preload="metadata"
          :src="redactedAudioSrc"
        ></audio>
      </div>
    </div>
    <a
      v-if="job?.status === 'done'"
      class="btn-export"
      :href="exportUrl(id)"
      download
    >⬇ Скачать всё (ZIP)</a>
  </section>

  <section v-if="job?.status === 'done'" class="card">
    <h2>Транскрипт</h2>
    <div class="tabs">
      <button
        class="tab"
        :class="{ active: activeTab === 'full' }"
        @click="activeTab = 'full'"
      >
        Исходный
      </button>
      <button
        class="tab"
        :class="{ active: activeTab === 'redacted' }"
        @click="activeTab = 'redacted'"
      >
        Обезличенный
      </button>
    </div>

    <div v-if="activeTranscript">
      <p class="muted" style="font-size: 13px; margin-top: 0">
        Язык: {{ activeTranscript.language }} · Длительность:
        {{ activeTranscript.duration.toFixed(2) }} с · PII:
        {{ activeTranscript.pii_count }}
      </p>
      <TranscriptWords
        :words="activeTranscript.words"
        :redacted="activeTab === 'redacted'"
      />
      <details style="margin-top: 16px">
        <summary class="muted">Показать сплошной текст</summary>
        <p class="mono" style="white-space: pre-wrap; margin-top: 8px">
          {{ activeTranscript.text }}
        </p>
      </details>
    </div>
    <p v-else class="muted">Загрузка транскрипта...</p>
  </section>

  <section v-if="job?.status === 'done'" class="card">
    <h2>События PII</h2>
    <p v-if="events.length" class="muted" style="font-size: 13px; margin-top: 0">
      Нажмите на строку, чтобы перейти к моменту в обезличенной записи.
    </p>
    <table v-if="events.length" class="tbl">
      <thead>
        <tr>
          <th>Интервал</th>
          <th>Тип</th>
          <th>Текст</th>
          <th>Источник</th>
          <th>Уверенность</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="ev in events"
          :key="ev.id"
          class="clickable"
          @click="seekRedacted(ev)"
        >
          <td class="mono">
            {{ formatSeconds(ev.start_sec) }} → {{ formatSeconds(ev.end_sec) }}
          </td>
          <td>
            <span :class="['pii', `pii-${ev.pii_type}`]">{{ ev.pii_type }}</span>
          </td>
          <td>{{ ev.text }}</td>
          <td>{{ ev.source ?? '—' }}</td>
          <td>{{ formatConfidence(ev.confidence) }}</td>
        </tr>
      </tbody>
    </table>
    <p v-else class="muted">Событий PII не обнаружено.</p>
  </section>
</template>

<style scoped>
.header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}
.header-row h2 {
  margin: 0;
}
.btn-export {
  display: inline-block;
  background: #2c7a4d;
  color: white;
  padding: 10px 18px;
  border-radius: 6px;
  text-decoration: none;
  font-weight: 600;
  margin: 12px 0;
}
.btn-export:hover { background: #245f3d; }
</style>
