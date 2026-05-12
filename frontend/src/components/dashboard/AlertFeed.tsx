import React from 'react';
import { useAppStore } from '../../store';

export function AlertFeed() {
  const { alerts, acknowledgeAlert } = useAppStore();
  const unacked = alerts.filter(a => !a.acknowledged);
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-white font-semibold">Live Alerts</h2>
        {unacked.length > 0 && (
          <span className="bg-red-900 text-red-200 text-xs px-2 py-1 rounded-full animate-pulse">
            {unacked.length} new
          </span>
        )}
      </div>
      {alerts.length === 0 ? (
        <div className="text-gray-500 text-center py-6">
          <div className="text-3xl mb-2">✅</div>
          <p className="text-sm">No active alerts</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {alerts.slice(0, 15).map((alert) => (
            <div
              key={alert.id}
              className={`p-3 rounded-lg border text-sm ${
                alert.acknowledged ? 'opacity-50' : ''
              } ${
                alert.severity === 'critical' ? 'bg-red-900/30 border-red-700 text-red-200' :
                alert.severity === 'warning' ? 'bg-yellow-900/30 border-yellow-700 text-yellow-200' :
                'bg-blue-900/30 border-blue-700 text-blue-200'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <span className="flex-1">{alert.message}</span>
                {!alert.acknowledged && (
                  <button
                    onClick={() => acknowledgeAlert(alert.id)}
                    className="text-xs opacity-70 hover:opacity-100 flex-shrink-0"
                  >✕</button>
                )}
              </div>
              <div className="text-xs opacity-60 mt-1">
                {new Date(alert.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
