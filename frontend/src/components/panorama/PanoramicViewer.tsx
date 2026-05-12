import React, { useRef, useState, useCallback, useEffect, Suspense } from 'react';

interface Props {
  panorama: { id: string; original_filename: string; camera_type: string; location_name?: string; floor_level?: number };
  overlayOpacity?: number;
  onPositionChange?: (yaw: number, pitch: number) => void;
}

// Lazy-load Three.js to avoid SSR issues
const ThreePanorama = React.lazy(() => import('./ThreePanorama'));

export function PanoramicViewer({ panorama, overlayOpacity = 0.5, onPositionChange }: Props) {
  const [fov, setFov] = useState(75);
  const handleWheel = useCallback((e: React.WheelEvent) => {
    setFov(f => Math.min(120, Math.max(30, f + e.deltaY * 0.05)));
  }, []);

  return (
    <div className="relative w-full h-full bg-black rounded-lg overflow-hidden" onWheel={handleWheel}>
      <Suspense fallback={<PanoramaPlaceholder name={panorama.original_filename} />}>
        <ThreePanorama panoramaId={panorama.id} fov={fov} onPositionChange={onPositionChange} />
      </Suspense>
      {/* HUD */}
      <div className="absolute top-4 left-4 pointer-events-none">
        <div className="bg-black/60 backdrop-blur-sm text-white text-xs px-3 py-2 rounded-lg space-y-1">
          <div className="font-semibold text-sm truncate max-w-[200px]">{panorama.original_filename}</div>
          <div className="text-gray-300">{panorama.camera_type} · FOV {fov.toFixed(0)}°</div>
          {panorama.location_name && <div className="text-blue-300">📍 {panorama.location_name}</div>}
        </div>
      </div>
      {/* Zoom Controls */}
      <div className="absolute bottom-4 right-4 flex gap-2">
        {[{ label: '+', action: () => setFov(f => Math.max(30, f - 10)) },
          { label: '−', action: () => setFov(f => Math.min(120, f + 10)) },
          { label: '⟳', action: () => setFov(75) }].map(({ label, action }) => (
          <button key={label} onClick={action}
            className="bg-black/60 text-white w-8 h-8 rounded-full flex items-center justify-center hover:bg-black/80">
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}

function PanoramaPlaceholder({ name }: { name: string }) {
  return (
    <div className="w-full h-full flex items-center justify-center bg-gray-900">
      <div className="text-center">
        <div className="text-6xl mb-3 animate-spin">🌐</div>
        <p className="text-gray-400 text-sm">Loading panorama...</p>
        <p className="text-gray-600 text-xs mt-1 truncate max-w-48">{name}</p>
      </div>
    </div>
  );
}

export default PanoramicViewer;
