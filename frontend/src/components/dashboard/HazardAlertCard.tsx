import React from 'react';
import type { Panorama } from '../../types';

export function HazardAlertCard({ panorama }: { panorama: Panorama | null }) {
  const hazards = panorama?.analysis_results?.hazards;
  const risk = hazards?.overall_risk_score ?? 0;
  const riskPct = (risk * 100).toFixed(0);
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-gray-400 text-sm font-medium">Hazard Risk</span>
        <span className="text-2xl">⚠️</span>
      </div>
      <div className={`text-3xl font-bold mb-1 ${risk >= 0.75 ? 'text-red-400' : risk >= 0.5 ? 'text-orange-400' : risk >= 0.25 ? 'text-yellow-400' : 'text-green-400'}`}>
        {riskPct}%
      </div>
      <div className="text-gray-400 text-sm">{hazards?.zone_count ?? 0} hazard zones</div>
      <div className="mt-3 pt-3 border-t border-gray-800">
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Workers at risk</span>
          <span className={`font-medium ${(hazards?.workers_in_hazard ?? 0) > 0 ? 'text-red-400' : 'text-green-400'}`}>
            {hazards?.workers_in_hazard ?? 0}
          </span>
        </div>
      </div>
    </div>
  );
}
