<script setup lang="ts">
import { computed } from 'vue'
import type { JobStatus } from '../types'

const props = defineProps<{ status: JobStatus }>()

const label = computed(() => {
  switch (props.status) {
    case 'queued':
      return 'в очереди'
    case 'running':
      return 'обработка'
    case 'done':
      return 'готово'
    case 'failed':
      return 'ошибка'
    default:
      return props.status
  }
})
</script>

<template>
  <span :class="['status', `status-${status}`]">{{ label }}</span>
</template>
