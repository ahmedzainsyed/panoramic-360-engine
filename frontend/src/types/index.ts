export interface Panorama {
  id: string; session_id: string; storage_key: string;
  original_filename: string; file_size_bytes: number;
  camera_type: string; format: string; width?: number; height?: number;
  location_name?: string; gps_latitude?: number; gps_longitude?: number;
  floor_level?: number; status: string; analysis_results?: AnalysisResults;
  created_at: string;
}
export interface AnalysisResults {
  segmentation?: { hazard_score: number; class_areas: Record<string,number>; inference_ms: number };
  detection?: { object_count: number; class_counts: Record<string,number>; worker_count: number };
  ppe?: { total_workers: number; compliance_rate: number; risk_level: string; alerts: string[]; violation_summary: Record<string,number> };
  hazards?: { zone_count: number; overall_risk_score: number; risk_level: string; alerts: string[]; workers_in_hazard: number };
  occupancy?: { spatial_utilization: number; activity_zone_count: number; worker_count_stats: Record<string,number> };
  navigation?: { accessibility_score: number; walkable_area_percent: number };
  total_processing_ms?: number;
}
export interface SiteSession { id: string; name: string; location_name?: string; panorama_count: number; is_active: boolean; created_at: string; }
export interface AlertItem { id: string; severity: string; message: string; panorama_id?: string; timestamp: string; acknowledged: boolean; }
