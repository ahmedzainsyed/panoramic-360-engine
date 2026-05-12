import React, { useMemo } from 'react';
import { useAppStore } from '../store';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

export function AnalyticsPage() {
  const { panoramas } = useAppStore();

  const timelineData = useMemo(() =>
    panoramas
      .filter(p => p.analysis_results?.ppe)
      .slice(0, 20)
      .map((p, i) => ({
        name: `Scan ${i + 1}`,
        compliance: parseFloat(((p.analysis_results!.ppe!.compliance_rate) * 100).toFixed(1)),
        risk: parseFloat(((p.analysis_results!.hazards?.overall_risk_score ?? 0) * 100).toFixed(1)),
        workers: p.analysis_results!.detection?.worker_count ?? 0,
      })), [panoramas]);

  const classData = useMemo(() => {
    if (!panoramas.length) return [];
    const latest = panoramas.find(p => p.analysis_results?.segmentation);
    if (!latest?.analysis_results?.segmentation) return [];
    return Object.entries(latest.analysis_results.segmentation.class_areas)
      .filter(([, v]) => v > 1.0)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 8)
      .map(([name, value]) => ({ name: name.replace('_', ' '), value: parseFloat(value.toFixed(1)) }));
  }, [panoramas]);

  const COLORS = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16'];

  return (
    <div className="p-6 bg-gray-950 min-h-screen text-white">
      <h1 className="text-2xl font-bold mb-6">Site Analytics</h1>

      {timelineData.length === 0 ? (
        <div className="text-center py-20 text-gray-500">
          <div className="text-5xl mb-4">📊</div>
          <p>No analyzed panoramas yet. Upload and analyze panoramas to see trends.</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* PPE Compliance Trend */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h2 className="text-white font-semibold mb-4">PPE Compliance & Hazard Risk Trend</h2>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} domain={[0, 100]} unit="%" />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                  labelStyle={{ color: '#f9fafb' }} />
                <Line type="monotone" dataKey="compliance" stroke="#10b981" strokeWidth={2} dot={false} name="PPE Compliance" />
                <Line type="monotone" dataKey="risk" stroke="#ef4444" strokeWidth={2} dot={false} name="Hazard Risk" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Worker Count */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h2 className="text-white font-semibold mb-4">Worker Count per Scan</h2>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
                <Bar dataKey="workers" name="Workers" radius={[4,4,0,0]}>
                  {timelineData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Segmentation Classes */}
          {classData.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <h2 className="text-white font-semibold mb-4">Scene Composition (Latest Scan)</h2>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={classData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 11 }} unit="%" />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} width={100} />
                  <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
                  <Bar dataKey="value" radius={[0,4,4,0]}>
                    {classData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
