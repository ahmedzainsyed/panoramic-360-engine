/**
 * API client hooks using TanStack Query.
 * Provides typed data fetching for all backend endpoints.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import type { Panorama, AnalysisResults, PPEResult, HazardResult } from '../types';
import { useAppStore } from '../store';

// Axios instance with auth interceptor
const api = axios.create({ baseURL: '/api/v1' });
api.interceptors.request.use((config) => {
  const token = useAppStore.getState().token;
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) useAppStore.getState().logout();
    return Promise.reject(err);
  }
);

// ── Panoramas ─────────────────────────────────────────────────
export const usePanoramas = (sessionId?: string) =>
  useQuery({
    queryKey: ['panoramas', sessionId],
    queryFn: async () => {
      const params = sessionId ? { session_id: sessionId } : {};
      const { data } = await api.get('/panoramas', { params });
      return data as { panoramas: Panorama[]; total: number };
    },
    staleTime: 30_000,
  });

export const usePanorama = (id: string) =>
  useQuery({
    queryKey: ['panorama', id],
    queryFn: async () => {
      const { data } = await api.get(`/panoramas/${id}`);
      return data as Panorama;
    },
    enabled: !!id,
  });

export const useAnalysisResults = (panoramaId: string) =>
  useQuery({
    queryKey: ['analysis', panoramaId],
    queryFn: async () => {
      const { data } = await api.get(`/analyze/${panoramaId}/results`);
      return data as AnalysisResults;
    },
    enabled: !!panoramaId,
    refetchInterval: (data) => data ? false : 5000, // poll until complete
  });

export const useAnalysisStatus = (panoramaId: string, enabled = true) =>
  useQuery({
    queryKey: ['analysis-status', panoramaId],
    queryFn: async () => {
      const { data } = await api.get(`/analyze/${panoramaId}/status`);
      return data;
    },
    enabled: enabled && !!panoramaId,
    refetchInterval: 3000,
  });

// ── PPE ───────────────────────────────────────────────────────
export const usePPEReport = (panoramaId: string) =>
  useQuery({
    queryKey: ['ppe', panoramaId],
    queryFn: async () => {
      const { data } = await api.get(`/ppe/${panoramaId}/report`);
      return data as PPEResult;
    },
    enabled: !!panoramaId,
  });

// ── Hazards ───────────────────────────────────────────────────
export const useHazardMap = (panoramaId: string) =>
  useQuery({
    queryKey: ['hazards', panoramaId],
    queryFn: async () => {
      const { data } = await api.get(`/hazards/${panoramaId}/map`);
      return data as HazardResult;
    },
    enabled: !!panoramaId,
  });

// ── Mutations ─────────────────────────────────────────────────
export const useRunAnalysis = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ panoramaId, modules }: { panoramaId: string; modules: string[] }) => {
      const { data } = await api.post('/analyze', { panorama_id: panoramaId, modules });
      return data;
    },
    onSuccess: (_, { panoramaId }) => {
      qc.invalidateQueries({ queryKey: ['analysis', panoramaId] });
    },
  });
};

export const useDeletePanorama = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => api.delete(`/panoramas/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['panoramas'] }),
  });
};

// ── Sessions ──────────────────────────────────────────────────
export const useSessions = () =>
  useQuery({
    queryKey: ['sessions'],
    queryFn: async () => {
      const { data } = await api.get('/sessions');
      return data;
    },
  });

export const useCreateSession = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (session: { id: string; name: string; location_name?: string }) => {
      const { data } = await api.post('/sessions', session);
      return data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['sessions'] }),
  });
};
