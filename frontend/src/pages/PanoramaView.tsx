import React, { useState, useCallback } from 'react';
import { useAppStore } from '../store';
import { PanoramicViewer } from '../components/panorama/PanoramicViewer';
import { AnalyticsPanel } from '../components/dashboard/AnalyticsPanel';

export function PanoramaView() {
  const { activePanorama, activeOverlay, setActiveOverlay } = useAppStore();
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [viewYaw, setViewYaw] = useState(0);
  const [viewPitch, setViewPitch] = useState(0);

  const handlePositionChange = useCallback((yaw: number, pitch: number) => {
    setViewYaw(yaw);
    setViewPitch(pitch);
  }, []);

  if (!activePanorama) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-950 text-gray-400">
        <div className="text-center">
          <div className="text-6xl mb-4">🌐</div>
          <h2 className="text-xl font-semibold text-white mb-2">No Panorama Selected</h2>
          <p className="text-sm">Select a panorama from the dashboard to view it here</p>
        </div>
      </div>
    );
  }

  const overlayButtons = [
    { key: 'segmentation', label: 'Segmentation', icon: '🎨' },
    { key: 'ppe', label: 'PPE', icon: '⛑️' },
    { key: 'hazards', label: 'Hazards', icon: '⚠️' },
    { key: 'occupancy', label: 'Occupancy', icon: '👥' },
    { key: 'navigation', label: 'Navigation', icon: '🗺️' },
  ];

  return (
    <div className="flex h-full bg-gray-950">
      {/* Main Viewer */}
      <div className="flex-1 flex flex-col">
        {/* Overlay toolbar */}
        <div className="flex items-center gap-2 p-3 bg-gray-900 border-b border-gray-800">
          <span className="text-gray-400 text-xs font-medium mr-2">OVERLAY:</span>
          {overlayButtons.map(({ key, label, icon }) => (
            <button
              key={key}
              onClick={() => setActiveOverlay(activeOverlay === key ? null : key)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeOverlay === key
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              <span>{icon}</span> {label}
            </button>
          ))}
          {activeOverlay && (
            <div className="ml-auto flex items-center gap-2">
              <span className="text-gray-400 text-xs">Opacity:</span>
              <input
                type="range" min={0} max={1} step={0.05}
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                className="w-24 accent-blue-500"
              />
              <span className="text-gray-400 text-xs w-8">{(overlayOpacity * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>

        {/* 360° Viewer */}
        <div className="flex-1 relative">
          <PanoramicViewer
            panorama={activePanorama}
            overlayOpacity={overlayOpacity}
            onPositionChange={handlePositionChange}
          />
          {/* Angular position indicator */}
          <div className="absolute bottom-4 left-4 bg-black/60 text-white text-xs px-3 py-1.5 rounded-lg font-mono">
            Yaw: {viewYaw.toFixed(1)}° · Pitch: {viewPitch.toFixed(1)}°
          </div>
        </div>
      </div>

      {/* Side Analytics Panel */}
      <div className="w-80 border-l border-gray-800 overflow-y-auto">
        <AnalyticsPanel panorama={activePanorama} />
      </div>
    </div>
  );
}
