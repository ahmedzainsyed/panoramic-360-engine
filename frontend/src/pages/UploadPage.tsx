import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import toast from 'react-hot-toast';
import { useAppStore } from '../store';

export function UploadPage() {
  const { token, addPanorama, addAlert } = useAppStore();
  const [uploads, setUploads] = useState<{ file: File; progress: number; status: string; id?: string }[]>([]);
  const [sessionId, setSessionId] = useState('site_' + Date.now());
  const [cameraType, setCameraType] = useState('unknown');

  const onDrop = useCallback(async (accepted: File[]) => {
    const newUploads = accepted.map(f => ({ file: f, progress: 0, status: 'queued' }));
    setUploads(prev => [...prev, ...newUploads]);

    for (let i = 0; i < accepted.length; i++) {
      const file = accepted[i];
      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', sessionId);
      formData.append('camera_type', cameraType);

      setUploads(prev => prev.map((u, idx) =>
        u.file === file ? { ...u, status: 'uploading', progress: 10 } : u
      ));

      try {
        const res = await axios.post('/api/v1/panoramas/upload', formData, {
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (e) => {
            const pct = Math.round((e.loaded / (e.total ?? 1)) * 80);
            setUploads(prev => prev.map(u => u.file === file ? { ...u, progress: pct } : u));
          },
        });
        setUploads(prev => prev.map(u =>
          u.file === file ? { ...u, status: 'processing', progress: 90, id: res.data.panorama_id } : u
        ));
        addPanorama({
          id: res.data.panorama_id,
          session_id: sessionId,
          storage_key: res.data.storage_key,
          original_filename: file.name,
          file_size_bytes: file.size,
          camera_type: cameraType,
          format: file.name.split('.').pop() ?? 'jpg',
          status: 'processing',
          created_at: new Date().toISOString(),
        });
        toast.success(`${file.name} uploaded successfully`);
        setTimeout(() => {
          setUploads(prev => prev.map(u => u.file === file ? { ...u, status: 'done', progress: 100 } : u));
        }, 1000);
      } catch (err: any) {
        setUploads(prev => prev.map(u =>
          u.file === file ? { ...u, status: 'error', progress: 0 } : u
        ));
        addAlert({ severity: 'warning', message: `Upload failed: ${file.name}` });
        toast.error(`Failed to upload ${file.name}`);
      }
    }
  }, [token, sessionId, cameraType, addPanorama, addAlert]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.tiff', '.tif'] },
    maxSize: 500 * 1024 * 1024,
  });

  return (
    <div className="p-6 bg-gray-950 min-h-screen">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-bold text-white mb-2">Upload Panoramas</h1>
        <p className="text-gray-400 text-sm mb-6">
          Upload 360° equirectangular panoramas from Insta360, Ricoh Theta, drones, or site scanners.
        </p>

        {/* Config */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="text-gray-400 text-xs mb-1 block">Session ID</label>
            <input value={sessionId} onChange={e => setSessionId(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500" />
          </div>
          <div>
            <label className="text-gray-400 text-xs mb-1 block">Camera Type</label>
            <select value={cameraType} onChange={e => setCameraType(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500">
              {['unknown', 'insta360', 'ricoh_theta', 'drone', 'matterport'].map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Dropzone */}
        <div {...getRootProps()} className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-colors ${
          isDragActive ? 'border-blue-500 bg-blue-900/10' : 'border-gray-700 hover:border-gray-500 bg-gray-900/40'
        }`}>
          <input {...getInputProps()} />
          <div className="text-6xl mb-4">{isDragActive ? '⬇️' : '🌐'}</div>
          <p className="text-white font-medium mb-1">
            {isDragActive ? 'Drop your panoramas here' : 'Drag & drop panoramas here'}
          </p>
          <p className="text-gray-400 text-sm">or click to browse · JPG, PNG, TIFF · max 500MB each</p>
          <div className="flex justify-center gap-4 mt-4 text-xs text-gray-600">
            {['Insta360', 'Ricoh Theta', 'DJI Drone', 'Matterport'].map(c => (
              <span key={c} className="bg-gray-800 px-2 py-1 rounded">{c}</span>
            ))}
          </div>
        </div>

        {/* Upload list */}
        {uploads.length > 0 && (
          <div className="mt-6 space-y-3">
            <h3 className="text-white font-medium text-sm">Uploads</h3>
            {uploads.map((u, i) => (
              <div key={i} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-lg">📷</span>
                    <div>
                      <div className="text-white text-sm font-medium truncate max-w-xs">{u.file.name}</div>
                      <div className="text-gray-500 text-xs">{(u.file.size / 1e6).toFixed(1)} MB</div>
                    </div>
                  </div>
                  <StatusBadge status={u.status} />
                </div>
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full transition-all duration-300 ${
                    u.status === 'error' ? 'bg-red-500' : u.status === 'done' ? 'bg-green-500' : 'bg-blue-500'
                  }`} style={{ width: `${u.progress}%` }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    queued: 'bg-gray-800 text-gray-400',
    uploading: 'bg-blue-900 text-blue-200',
    processing: 'bg-yellow-900 text-yellow-200',
    done: 'bg-green-900 text-green-200',
    error: 'bg-red-900 text-red-200',
  };
  return <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${map[status] ?? 'bg-gray-800 text-gray-400'}`}>{status}</span>;
}
