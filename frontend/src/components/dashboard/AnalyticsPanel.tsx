import React from 'react';
import type { Panorama } from '../../types';

interface Props { panorama: Panorama; }

export function AnalyticsPanel({ panorama }: Props) {
  const r = panorama.analysis_results;

  return (
    <div className="p-4 space-y-4 bg-gray-950">
      <h3 className="text-white font-semibold text-sm uppercase tracking-wide">Analysis Results</h3>

      {/* PPE Module */}
      {r?.ppe && (
        <ModuleCard title="PPE Compliance" icon="⛑️"
          risk={r.ppe.risk_level} items={[
            { label: 'Workers', value: String(r.ppe.total_workers) },
            { label: 'Compliance', value: `${(r.ppe.compliance_rate * 100).toFixed(0)}%` },
            { label: 'Violations', value: String(Object.values(r.ppe.violation_summary).reduce((a,b) => a+b, 0)) },
          ]} alerts={r.ppe.alerts} />
      )}

      {/* Hazards Module */}
      {r?.hazards && (
        <ModuleCard title="Hazard Detection" icon="⚠️"
          risk={r.hazards.risk_level} items={[
            { label: 'Zones', value: String(r.hazards.zone_count) },
            { label: 'Risk Score', value: `${(r.hazards.overall_risk_score * 100).toFixed(0)}%` },
            { label: 'Workers at Risk', value: String(r.hazards.workers_in_hazard) },
          ]} alerts={r.hazards.alerts} />
      )}

      {/* Segmentation Module */}
      {r?.segmentation && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">🎨</span>
            <span className="text-white text-sm font-medium">Segmentation</span>
          </div>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {Object.entries(r.segmentation.class_areas)
              .filter(([, v]) => v > 0.5)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 8)
              .map(([cls, pct]) => (
                <div key={cls} className="flex items-center gap-2">
                  <div className="flex-1 text-gray-400 text-xs capitalize">{cls.replace('_', ' ')}</div>
                  <div className="text-gray-300 text-xs w-12 text-right">{pct.toFixed(1)}%</div>
                  <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: `${Math.min(pct, 100)}%` }} />
                  </div>
                </div>
              ))}
          </div>
          <div className="mt-2 pt-2 border-t border-gray-800 text-xs text-gray-500">
            Hazard score: {(r.segmentation.hazard_score * 100).toFixed(0)}%
            · {r.segmentation.inference_ms.toFixed(0)}ms
          </div>
        </div>
      )}

      {/* Occupancy Module */}
      {r?.occupancy && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">👥</span>
            <span className="text-white text-sm font-medium">Occupancy</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Utilization" value={`${(r.occupancy.spatial_utilization * 100).toFixed(1)}%`} />
            <Stat label="Activity Zones" value={String(r.occupancy.activity_zone_count)} />
          </div>
        </div>
      )}

      {/* Navigation Module */}
      {r?.navigation && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">🗺️</span>
            <span className="text-white text-sm font-medium">Navigation</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Accessibility" value={`${(r.navigation.accessibility_score * 100).toFixed(0)}%`} />
            <Stat label="Walkable Area" value={`${r.navigation.walkable_area_percent.toFixed(1)}%`} />
          </div>
        </div>
      )}

      {/* Processing time */}
      {r?.total_processing_ms && (
        <div className="text-center text-gray-600 text-xs">
          Total processing: {(r.total_processing_ms / 1000).toFixed(1)}s
        </div>
      )}
    </div>
  );
}

function ModuleCard({ title, icon, risk, items, alerts }: {
  title: string; icon: string; risk: string;
  items: { label: string; value: string }[];
  alerts?: string[];
}) {
  const riskColor = risk === 'critical' ? 'border-red-700' : risk === 'high' ? 'border-orange-700' :
    risk === 'medium' ? 'border-yellow-700' : 'border-green-800';
  return (
    <div className={`bg-gray-900 border rounded-xl p-4 ${riskColor}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-lg">{icon}</span>
          <span className="text-white text-sm font-medium">{title}</span>
        </div>
        <RiskBadge level={risk} />
      </div>
      <div className="grid grid-cols-3 gap-2 mb-3">
        {items.map(({ label, value }) => (
          <Stat key={label} label={label} value={value} />
        ))}
      </div>
      {alerts && alerts.length > 0 && (
        <div className="space-y-1 mt-2 pt-2 border-t border-gray-800">
          {alerts.slice(0, 2).map((a, i) => (
            <p key={i} className="text-xs text-orange-300 leading-tight">{a}</p>
          ))}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <div className="text-white font-bold text-base">{value}</div>
      <div className="text-gray-500 text-xs mt-0.5">{label}</div>
    </div>
  );
}

function RiskBadge({ level }: { level: string }) {
  const cls = level === 'critical' ? 'bg-red-900 text-red-200' :
    level === 'high' ? 'bg-orange-900 text-orange-200' :
    level === 'medium' ? 'bg-yellow-900 text-yellow-200' :
    'bg-green-900 text-green-200';
  return <span className={`${cls} text-xs px-2 py-0.5 rounded-full font-medium capitalize`}>{level}</span>;
}
