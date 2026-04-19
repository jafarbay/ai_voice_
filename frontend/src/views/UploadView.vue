<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { uploadJob, ApiError } from '../api'

const ALLOWED = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.opus', '.webm']
const MAX_MB = 200

const router = useRouter()
const file = ref<File | null>(null)
const dragOver = ref(false)
const uploading = ref(false)
const error = ref<string | null>(null)
const inputEl = ref<HTMLInputElement | null>(null)

const filename = computed(() => file.value?.name ?? '')

function extOf(name: string): string {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.substring(i).toLowerCase() : ''
}

function validate(f: File): string | null {
  const ext = extOf(f.name)
  if (!ALLOWED.includes(ext)) {
    return `Недопустимое расширение '${ext}'. Разрешено: ${ALLOWED.join(', ')}`
  }
  if (f.size > MAX_MB * 1024 * 1024) {
    return `Файл больше ${MAX_MB} МБ`
  }
  if (f.size === 0) {
    return 'Файл пустой'
  }
  return null
}

function pickFile(f: File | null | undefined) {
  if (!f) return
  const err = validate(f)
  if (err) {
    error.value = err
    file.value = null
    return
  }
  error.value = null
  file.value = f
}

function onInputChange(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0]
  pickFile(f)
}

function onDrop(e: DragEvent) {
  e.preventDefault()
  dragOver.value = false
  const f = e.dataTransfer?.files?.[0]
  pickFile(f)
}

function onDragOver(e: DragEvent) {
  e.preventDefault()
  dragOver.value = true
}

function onDragLeave() {
  dragOver.value = false
}

async function submit() {
  if (!file.value || uploading.value) return
  uploading.value = true
  error.value = null
  try {
    const resp = await uploadJob(file.value)
    router.push({ name: 'job-detail', params: { id: resp.job_id } })
  } catch (e) {
    error.value =
      e instanceof ApiError ? e.detail : (e as Error)?.message ?? 'Ошибка загрузки'
  } finally {
    uploading.value = false
  }
}

function resetFile() {
  file.value = null
  if (inputEl.value) inputEl.value.value = ''
}
</script>

<template>
  <section class="card">
    <h2>Загрузка аудиофайла</h2>
    <p class="muted">
      Загрузите аудиозапись — она будет расшифрована и обезличена. Поддерживаются
      форматы: {{ ALLOWED.join(', ') }}. Размер до {{ MAX_MB }} МБ.
    </p>

    <div v-if="error" class="error-box">{{ error }}</div>

    <div
      :class="['dropzone', { over: dragOver, filled: !!file }]"
      @dragover="onDragOver"
      @dragleave="onDragLeave"
      @drop="onDrop"
      @click="inputEl?.click()"
    >
      <input
        ref="inputEl"
        type="file"
        :accept="ALLOWED.join(',')"
        hidden
        @change="onInputChange"
      />
      <div v-if="!file">
        <div class="dropzone-title">Перетащите файл сюда</div>
        <div class="muted">или нажмите, чтобы выбрать</div>
      </div>
      <div v-else class="picked">
        <div class="dropzone-title">{{ filename }}</div>
        <div class="muted">
          {{ (file.size / (1024 * 1024)).toFixed(2) }} МБ
        </div>
      </div>
    </div>

    <div class="actions">
      <button
        class="btn btn-primary"
        :disabled="!file || uploading"
        @click="submit"
      >
        {{ uploading ? 'Загрузка...' : 'Загрузить и обработать' }}
      </button>
      <button
        v-if="file && !uploading"
        class="btn"
        type="button"
        @click="resetFile"
      >
        Сбросить
      </button>
    </div>
  </section>
</template>

<style scoped>
.dropzone {
  border: 2px dashed #cbd5e1;
  border-radius: 10px;
  padding: 32px 16px;
  text-align: center;
  background: #fafbfc;
  cursor: pointer;
  transition: border-color 0.15s, background-color 0.15s;
  margin-bottom: 16px;
}
.dropzone:hover,
.dropzone.over {
  border-color: #2563eb;
  background: #eff6ff;
}
.dropzone.filled {
  border-style: solid;
  border-color: #86efac;
  background: #f0fdf4;
}
.dropzone-title {
  font-weight: 600;
  font-size: 15px;
  margin-bottom: 4px;
  word-break: break-all;
}
.actions {
  display: flex;
  gap: 8px;
}
</style>
