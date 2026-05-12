import React, { useEffect } from 'react';
import { useAppStore } from '../store';
import { SiteOverviewCard } from '../components/dashboard/SiteOverviewCard';
import { PPEComplianceCard } from '../components/dashboard/PPEComplianceCard';
import { HazardAlertCard } from '../components/dashboard/HazardAlertCard';
import { OccupancyCard } from '../components/dashboard/OccupancyCard';
import { PanoramaList } from '../components/dashboard/PanoramaList';
import { AlertFeed } from '../components/dashboard/AlertFeed';

export function Dashboard() {
  const { activePanorama, activeSession, alerts } = useAppStore();

  return (
    <div className="flex flex-col gap-6 p-6 bg-gray-950 min-h-screen text-white">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">360° Site Intelligence</h1>
          <p className="text-gray-400 text-sm mt-1">
            {activeSession ? `Session: ${activeSession.name}` : 'No active session'}
            {activePanorama ? ` · ${activePanorama.original_filename}` : ''}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
            alerts.some(a => a.severity === 'critical' && !a.acknowledged)
              ? 'bg-red-900 text-red-200 animate-pulse'
              : 'bg-green-900 text-green-200'
          }`}>
            {alerts.filter(a => !a.acknowledged).length > 0
              ? `${alerts.filter(a => !a.acknowledged).length} Active Alerts`
              : 'All Clear'}
          </span>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <SiteOverviewCard panorama={activePanorama} />
        <PPEComplianceCard panorama={activePanorama} />
        <HazardAlertCard panorama={activePanorama} />
        <OccupancyCard panorama={activePanorama} />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2">
          <PanoramaList />
        </div>
        <div>
          <AlertFeed />
        </div>
      </div>
    </div>
  );
}
