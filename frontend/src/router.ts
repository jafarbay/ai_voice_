import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'upload',
    component: () => import('./views/UploadView.vue'),
  },
  {
    path: '/jobs',
    name: 'jobs',
    component: () => import('./views/JobsView.vue'),
  },
  {
    path: '/jobs/:id',
    name: 'job-detail',
    component: () => import('./views/JobDetailView.vue'),
    props: true,
  },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
