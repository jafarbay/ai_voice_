<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <h1>Анонимизация голоса</h1>
      <nav class="app-nav">
        <RouterLink to="/">Загрузить</RouterLink>
        <RouterLink to="/jobs">Задачи</RouterLink>
      </nav>
    </header>
    <main>
      <RouterView />
    </main>
  </div>
</template>
