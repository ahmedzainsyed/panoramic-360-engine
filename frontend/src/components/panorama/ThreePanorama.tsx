import React, { useRef, useEffect, useCallback } from 'react';
import * as THREE from 'three';

interface Props {
  panoramaId: string;
  fov: number;
  onPositionChange?: (yaw: number, pitch: number) => void;
}

export default function ThreePanorama({ panoramaId, fov, onPositionChange }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    sphere: THREE.Mesh;
    animId: number;
    isDragging: boolean;
    lastMouse: { x: number; y: number };
    yaw: number;
    pitch: number;
  } | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;
    const el = mountRef.current;
    const w = el.clientWidth, h = el.clientHeight;

    // Setup renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    el.appendChild(renderer.domElement);

    // Scene + camera
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(fov, w / h, 0.1, 1000);
    camera.position.set(0, 0, 0.001);

    // Sphere with checkerboard texture (placeholder for actual panorama)
    const geometry = new THREE.SphereGeometry(500, 128, 64);
    const canvas2d = document.createElement('canvas');
    canvas2d.width = 512; canvas2d.height = 256;
    const ctx = canvas2d.getContext('2d')!;
    // Draw placeholder gradient
    const grad = ctx.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, '#1a1a2e'); grad.addColorStop(0.5, '#16213e'); grad.addColorStop(1, '#0f3460');
    ctx.fillStyle = grad; ctx.fillRect(0, 0, 512, 256);
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    for (let i = 0; i < 512; i += 32) for (let j = 0; j < 256; j += 32)
      if ((i/32 + j/32) % 2 === 0) ctx.fillRect(i, j, 32, 32);
    ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.font = '20px monospace';
    ctx.fillText('360° Panorama - ' + panoramaId.slice(0,8), 160, 128);
    const tex = new THREE.CanvasTexture(canvas2d);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    const material = new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide });
    const sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);

    // Try to load actual panorama texture
    const loader = new THREE.TextureLoader();
    loader.load(
      `/api/v1/panoramas/${panoramaId}/thumbnail?size=2048`,
      (loadedTex) => {
        loadedTex.mapping = THREE.EquirectangularReflectionMapping;
        loadedTex.colorSpace = THREE.SRGBColorSpace;
        loadedTex.wrapS = THREE.RepeatWrapping;
        loadedTex.repeat.x = -1;
        (sphere.material as THREE.MeshBasicMaterial).map = loadedTex;
        (sphere.material as THREE.MeshBasicMaterial).needsUpdate = true;
      },
      undefined,
      () => {} // Ignore load errors, keep placeholder
    );

    let isDragging = false, lastMouse = { x: 0, y: 0 }, yaw = 0, pitch = 0;

    const onMouseDown = (e: MouseEvent) => { isDragging = true; lastMouse = { x: e.clientX, y: e.clientY }; };
    const onMouseUp = () => { isDragging = false; };
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - lastMouse.x, dy = e.clientY - lastMouse.y;
      yaw -= dx * 0.2; pitch = Math.max(-85, Math.min(85, pitch - dy * 0.2));
      lastMouse = { x: e.clientX, y: e.clientY };
      onPositionChange?.(yaw, pitch);
    };
    const onTouchStart = (e: TouchEvent) => { isDragging = true; lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY }; };
    const onTouchMove = (e: TouchEvent) => {
      if (!isDragging) return;
      const dx = e.touches[0].clientX - lastMouse.x, dy = e.touches[0].clientY - lastMouse.y;
      yaw -= dx * 0.2; pitch = Math.max(-85, Math.min(85, pitch - dy * 0.2));
      lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    };

    renderer.domElement.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('touchstart', onTouchStart, { passive: true });
    renderer.domElement.addEventListener('touchend', onMouseUp);
    renderer.domElement.addEventListener('touchmove', onTouchMove, { passive: true });

    // Animation loop
    let animId = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      const phi = THREE.MathUtils.degToRad(90 - pitch);
      const theta = THREE.MathUtils.degToRad(yaw);
      camera.lookAt(
        Math.sin(phi) * Math.cos(theta),
        Math.cos(phi),
        Math.sin(phi) * Math.sin(theta)
      );
      camera.fov = fov;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    };
    animate();

    // Resize observer
    const ro = new ResizeObserver(() => {
      const nw = el.clientWidth, nh = el.clientHeight;
      renderer.setSize(nw, nh);
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
    });
    ro.observe(el);

    sceneRef.current = { renderer, scene, camera, sphere, animId, isDragging, lastMouse, yaw, pitch };

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
      renderer.domElement.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('mousemove', onMouseMove);
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, [panoramaId]);

  // Update FOV reactively
  useEffect(() => {
    if (sceneRef.current) {
      sceneRef.current.camera.fov = fov;
      sceneRef.current.camera.updateProjectionMatrix();
    }
  }, [fov]);

  return <div ref={mountRef} className="w-full h-full" style={{ cursor: 'grab' }} />;
}
