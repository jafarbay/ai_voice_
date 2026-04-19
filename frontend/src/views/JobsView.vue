<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ApiError, listJobs } from '../api'
import type { JobInfo } from '../types'
import StatusBadge from '../components/StatusBadge.vue'

const jobs = ref<JobInfo[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

async function refresh() {
  loading.value = true
  error.value = null
  try {
    jobs.value = await listJobs()
  } catch (e) {
    error.value =
      e instanceof ApiError ? e.detail : (e as Error)?.message ?? 'Ошибка'
  } finally {
    loading.value = false
  }
}

function formatTime(ts: string): string {
  const d = new Date(ts)
  if (isNaN(d.getTime())) return ts
  return d.toLocaleString('ru-RU')
}

onMounted(refresh)
</script>

<template>
  <section class="card">
    <div class="header-row">
      <h2>Последние задачи</h2>
      <button class="btn" :disabled="loading" @click="refresh">
        {{ loading ? 'Обновление...' : 'Обновить' }}
      </button>
    </div>

    <div v-if="error" class="error-box">{{ error }}</div>

    <table v-if="jobs.length" class="tbl">
      <thead>
        <tr>
          <th>Создано</th>
          <th>Файл</th>
          <th>Статус</th>
          <th>PII</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="job in jobs" :key="job.id">
          <td class="mono">{{ formatTime(job.created_at) }}</td>
          <td>{{ job.input_filename ?? '—' }}</td>
          <td><StatusBadge :status="job.status" /></td>
          <td>{{ job.pii_count ?? '—' }}</td>
          <td>
            <RouterLink :to="{ name: 'job-detail', params: { id: job.id } }">
              Открыть
            </RouterLink>
          </td>
        </tr>
      </tbody>
    </table>
    <p v-else-if="!loading" class="muted">Пока нет задач.</p>
  </section>
</template>

<style scoped>
.header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
.header-row h2 {
  margin: 0;
}
</style>
