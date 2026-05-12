import React from 'react';
import { useAppStore } from '../../store';

export function PanoramaList() {
  const { panoramas, setActivePanorama, activePanorama } = useAppStore();
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <h2 className="text-white font-semibold mb-4">Recent Panoramas</h2>
      {panoramas.length === 0 ? (
        <div className="text-gray-500 text-center py-8">
          <div className="text-4xl mb-2">🌐</div>
          <p>No panoramas uploaded yet</p>
          <p className="text-sm mt-1">Upload a 360° image to get started</p>
        </div>
      ) : (
        <div className="space-y-2">
          {panoramas.slice(0, 10).map((p) => (
            <div
              key={p.id}
              onClick={() => setActivePanorama(p)}
              className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                activePanorama?.id === p.id ? 'bg-blue-900/40 border border-blue-700' : 'hover:bg-gray-800'
              }`}
            >
              <div className="w-10 h-10 bg-gray-700 rounded-lg flex items-center justify-center text-lg flex-shrink-0">
                {p.camera_type === 'drone' ? '🚁' : p.camera_type === 'insta360' ? '📷' : '🌐'}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-white text-sm font-medium truncate">{p.original_filename}</div>
                <div className="text-gray-400 text-xs mt-0.5">{p.camera_type} · {p.status}</div>
              </div>
              <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                p.status === 'completed' ? 'bg-green-400' :
                p.status === 'processing' ? 'bg-yellow-400 animate-pulse' :
                p.status === 'failed' ? 'bg-red-400' : 'bg-gray-500'
              }`} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
