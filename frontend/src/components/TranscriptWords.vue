<script setup lang="ts">
import { computed } from 'vue'
import type { Word } from '../types'

const props = defineProps<{
  words: Word[]
  redacted: boolean
}>()

interface Turn {
  speaker: string | null
  words: Word[]
}

const turns = computed<Turn[]>(() => {
  const out: Turn[] = []
  for (const w of props.words) {
    const sp = w.speaker ?? null
    const last = out[out.length - 1]
    if (last && last.speaker === sp) {
      last.words.push(w)
    } else {
      out.push({ speaker: sp, words: [w] })
    }
  }
  return out
})

function speakerLabel(sp: string | null): string {
  if (!sp) return ''
  // "SPEAKER_00" → "Спикер 1", "SPEAKER_01" → "Спикер 2"
  const m = /(\d+)/.exec(sp)
  if (m) return `Спикер ${parseInt(m[1], 10) + 1}`
  return sp
}
</script>

<template>
  <div class="word-container">
    <div v-for="(turn, ti) in turns" :key="ti" class="turn">
      <span v-if="turn.speaker" class="speaker-label">{{ speakerLabel(turn.speaker) }}:</span>
      <template v-for="(w, i) in turn.words" :key="i">
        <span
          v-if="w.pii_type"
          :class="['word', 'pii', `pii-${w.pii_type}`]"
          :title="redacted && w.original_word ? `было: ${w.original_word}` : w.pii_type"
        >{{ w.word }}</span>
        <span v-else class="word">{{ w.word }}</span>
      </template>
    </div>
  </div>
</template>

<style scoped>
.turn {
  margin-bottom: 8px;
  line-height: 1.6;
}
.speaker-label {
  display: inline-block;
  font-weight: 700;
  color: #555;
  margin-right: 6px;
  font-size: 0.9em;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
</style>
