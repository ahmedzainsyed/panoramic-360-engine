import React from 'react';
import type { Panorama } from '../../types';

export function OccupancyCard({ panorama }: { panorama: Panorama | null }) {
  const occ = panorama?.analysis_results?.occupancy;
  const util = ((occ?.spatial_utilization ?? 0) * 100).toFixed(1);
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-gray-400 text-sm font-medium">Site Utilization</span>
        <span className="text-2xl">📊</span>
      </div>
      <div className="text-3xl font-bold text-blue-400 mb-1">{util}%</div>
      <div className="text-gray-400 text-sm">Space utilized</div>
      <div className="mt-3 pt-3 border-t border-gray-800">
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Activity zones</span>
          <span className="text-blue-400 font-medium">{occ?.activity_zone_count ?? 0}</span>
        </div>
      </div>
    </div>
  );
}
