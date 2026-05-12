import React from 'react';
import type { Panorama } from '../../types';

export function SiteOverviewCard({ panorama }: { panorama: Panorama | null }) {
  const results = panorama?.analysis_results;
  const workerCount = results?.detection?.worker_count ?? 0;
  const totalObjects = results?.detection?.object_count ?? 0;
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-gray-400 text-sm font-medium">Site Overview</span>
        <span className="text-2xl">🏗️</span>
      </div>
      <div className="text-3xl font-bold text-white mb-1">{workerCount}</div>
      <div className="text-gray-400 text-sm">Workers on site</div>
      <div className="mt-3 pt-3 border-t border-gray-800">
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Objects detected</span>
          <span className="text-blue-400 font-medium">{totalObjects}</span>
        </div>
      </div>
    </div>
  );
}
