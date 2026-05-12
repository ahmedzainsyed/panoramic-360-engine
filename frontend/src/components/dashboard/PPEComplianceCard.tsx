import React from 'react';
import type { Panorama } from '../../types';

export function PPEComplianceCard({ panorama }: { panorama: Panorama | null }) {
  const ppe = panorama?.analysis_results?.ppe;
  const rate = ppe?.compliance_rate ?? 1.0;
  const pct = (rate * 100).toFixed(0);
  const riskColors: Record<string, string> = {
    low: 'text-green-400', medium: 'text-yellow-400', high: 'text-orange-400', critical: 'text-red-400'
  };
  const color = riskColors[ppe?.risk_level ?? 'low'] ?? 'text-green-400';
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-gray-400 text-sm font-medium">PPE Compliance</span>
        <span className="text-2xl">⛑️</span>
      </div>
      <div className={`text-3xl font-bold mb-1 ${color}`}>{pct}%</div>
      <div className="text-gray-400 text-sm">{ppe?.total_workers ?? 0} workers assessed</div>
      <div className="mt-3">
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <div className={`h-full rounded-full transition-all duration-500 ${
            rate >= 0.9 ? 'bg-green-500' : rate >= 0.75 ? 'bg-yellow-500' : 'bg-red-500'
          }`} style={{ width: `${pct}%` }} />
        </div>
      </div>
      {ppe?.risk_level && (
        <div className={`mt-2 text-xs font-medium uppercase ${color}`}>
          Risk: {ppe.risk_level}
        </div>
      )}
    </div>
  );
}
