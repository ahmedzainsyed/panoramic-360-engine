import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { Panorama, SiteSession, AlertItem } from '../types';

interface AppState {
  token: string | null; userEmail: string | null;
  setToken: (t: string | null, e?: string) => void;
  logout: () => void;
  activePanorama: Panorama | null;
  setActivePanorama: (p: Panorama | null) => void;
  activeSession: SiteSession | null;
  setActiveSession: (s: SiteSession | null) => void;
  panoramas: Panorama[];
  setPanoramas: (ps: Panorama[]) => void;
  addPanorama: (p: Panorama) => void;
  alerts: AlertItem[];
  addAlert: (a: Omit<AlertItem, 'id'|'timestamp'|'acknowledged'>) => void;
  sidebarOpen: boolean; toggleSidebar: () => void;
  activeOverlay: string | null; setActiveOverlay: (o: string | null) => void;
}

export const useAppStore = create<AppState>()(
  devtools(persist((set) => ({
    token: null, userEmail: null,
    setToken: (token, email) => set({ token, userEmail: email ?? null }),
    logout: () => set({ token: null, userEmail: null }),
    activePanorama: null, setActivePanorama: (p) => set({ activePanorama: p }),
    activeSession: null, setActiveSession: (s) => set({ activeSession: s }),
    panoramas: [], setPanoramas: (ps) => set({ panoramas: ps }),
    addPanorama: (p) => set((s) => ({ panoramas: [p, ...s.panoramas] })),
    alerts: [],
    addAlert: (a) => set((s) => ({ alerts: [{ ...a, id: Math.random().toString(36).slice(2), timestamp: new Date().toISOString(), acknowledged: false }, ...s.alerts].slice(0, 50) })),
    sidebarOpen: true, toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
    activeOverlay: null, setActiveOverlay: (o) => set({ activeOverlay: o }),
  }), { name: 'panoramic-store', partialize: (s) => ({ token: s.token, userEmail: s.userEmail }) }))
);
