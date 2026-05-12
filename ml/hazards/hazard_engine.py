"""Hazard Zone Segmentation Engine"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import structlog
logger = structlog.get_logger(__name__)

class HazardType(str, Enum):
    OPEN_SHAFT="open_shaft"; UNSAFE_EDGE="unsafe_edge"; RESTRICTED_ZONE="restricted_zone"
    MACHINERY_ZONE="machinery_zone"; ACTIVE_WORK_ZONE="active_work_zone"; FALL_HAZARD="fall_hazard"

HAZARD_SEVERITY = {HazardType.OPEN_SHAFT:1.0,HazardType.UNSAFE_EDGE:0.9,HazardType.FALL_HAZARD:0.85,
    HazardType.MACHINERY_ZONE:0.70,HazardType.RESTRICTED_ZONE:0.55,HazardType.ACTIVE_WORK_ZONE:0.40}
SEG_CLASS_TO_HAZARD = {20:HazardType.OPEN_SHAFT,13:HazardType.UNSAFE_EDGE,
    19:HazardType.RESTRICTED_ZONE,12:HazardType.RESTRICTED_ZONE}

@dataclass
class HazardZone:
    zone_id:str; hazard_type:HazardType; severity:float; bbox:Tuple[int,int,int,int]
    mask:Optional[np.ndarray]=None; area_pixels:int=0; area_percent:float=0.0
    centroid_x:float=0.0; centroid_y:float=0.0; panorama_yaw:float=0.0
    confidence:float=1.0; worker_proximity_count:int=0; description:str=""

@dataclass
class HazardAnalysisResult:
    panorama_id:str; zones:List[HazardZone]; risk_map:np.ndarray
    overall_risk_score:float; risk_level:str; zone_count_by_type:Dict[str,int]
    alerts:List[str]; worker_in_hazard_count:int; inference_time_ms:float=0.0
    @property
    def has_critical_hazards(self)->bool: return any(z.severity>=0.85 for z in self.zones)

class HazardZoneEngine:
    def __init__(self,segmentation_model=None,device="cuda",min_hazard_area_pixels=500,proximity_threshold_pixels=150):
        self.segmentation_model=segmentation_model; self.device=device
        self.min_hazard_area_pixels=min_hazard_area_pixels; self.proximity_threshold=proximity_threshold_pixels

    def analyze_panorama(self,image,panorama_id,seg_result=None,worker_boxes=None):
        t0=time.perf_counter(); h,w=image.shape[:2]
        if seg_result is None and self.segmentation_model is not None:
            seg_result=self.segmentation_model.segment_panorama(image)
        seg_zones=self._extract_zones_from_segmentation(seg_result,(h,w)) if seg_result else []
        edge_zones=self._detect_edge_hazards(image)
        all_zones=self._merge_zones(seg_zones+edge_zones)
        risk_map=self._build_risk_map(all_zones,h,w)
        if worker_boxes is not None and len(worker_boxes)>0:
            all_zones=self._analyze_worker_proximity(all_zones,worker_boxes)
        overall_risk=self._compute_overall_risk(all_zones,risk_map)
        risk_level=self._risk_score_to_level(overall_risk)
        alerts=self._generate_alerts(all_zones,overall_risk)
        zone_count_by_type:Dict[str,int]={}
        for z in all_zones: zone_count_by_type[z.hazard_type.value]=zone_count_by_type.get(z.hazard_type.value,0)+1
        elapsed_ms=(time.perf_counter()-t0)*1000
        logger.info("hazard_analysis_complete",panorama_id=panorama_id,zones=len(all_zones),risk=f"{overall_risk:.3f}",ms=f"{elapsed_ms:.1f}")
        return HazardAnalysisResult(panorama_id=panorama_id,zones=all_zones,risk_map=risk_map,
            overall_risk_score=overall_risk,risk_level=risk_level,zone_count_by_type=zone_count_by_type,
            alerts=alerts,worker_in_hazard_count=sum(z.worker_proximity_count for z in all_zones),
            inference_time_ms=elapsed_ms)

    def _extract_zones_from_segmentation(self,seg_result,image_shape):
        h,w=image_shape; zones=[]
        for class_id,hazard_type in SEG_CLASS_TO_HAZARD.items():
            class_mask=(seg_result.semantic_mask==class_id).astype(np.uint8)
            if class_mask.sum()<self.min_hazard_area_pixels: continue
            num_labels,labels,stats,centroids=cv2.connectedComponentsWithStats(class_mask,8)
            for label_id in range(1,num_labels):
                area=int(stats[label_id,cv2.CC_STAT_AREA])
                if area<self.min_hazard_area_pixels: continue
                x1=int(stats[label_id,cv2.CC_STAT_LEFT]); y1=int(stats[label_id,cv2.CC_STAT_TOP])
                bw=int(stats[label_id,cv2.CC_STAT_WIDTH]); bh=int(stats[label_id,cv2.CC_STAT_HEIGHT])
                cx,cy=float(centroids[label_id,0]),float(centroids[label_id,1])
                zone_mask=(labels==label_id); severity=HAZARD_SEVERITY.get(hazard_type,0.5)
                zones.append(HazardZone(zone_id=f"seg_{class_id}_{label_id}",hazard_type=hazard_type,
                    severity=severity,bbox=(x1,y1,x1+bw,y1+bh),mask=zone_mask,area_pixels=area,
                    area_percent=area/(h*w)*100.0,centroid_x=cx,centroid_y=cy,
                    panorama_yaw=(cx/w)*360.0,confidence=1.0,description=f"Seg-detected {hazard_type.value}"))
        return zones

    def _detect_edge_hazards(self,image):
        h,w=image.shape[:2]
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) if image.ndim==3 else image
        floor_region=gray[h//2:,:]
        edges=cv2.Canny(floor_region,50,150)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        edges_dilated=cv2.dilate(edges,kernel,iterations=3)
        contours,_=cv2.findContours(edges_dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        zones=[]
        for i,contour in enumerate(contours):
            area=cv2.contourArea(contour)
            if area<self.min_hazard_area_pixels*0.5: continue
            x,y,bw,bh=cv2.boundingRect(contour); y+=h//2
            M=cv2.moments(contour)
            if M["m00"]==0: continue
            cx=M["m10"]/M["m00"]; cy=M["m01"]/M["m00"]+h//2
            zones.append(HazardZone(zone_id=f"edge_{i}",hazard_type=HazardType.UNSAFE_EDGE,
                severity=0.45,bbox=(x,y,x+bw,y+bh),area_pixels=int(area),
                area_percent=area/(h*w)*100.0,centroid_x=cx,centroid_y=cy,
                panorama_yaw=(cx/w)*360.0,confidence=0.6,description="Edge-detected drop hazard"))
        return zones

    def _merge_zones(self,zones):
        if not zones: return []
        zones_sorted=sorted(zones,key=lambda z:z.severity,reverse=True)
        kept=[]
        for zone in zones_sorted:
            suppressed=False
            for kz in kept:
                if kz.hazard_type!=zone.hazard_type: continue
                if self._bbox_iou(zone.bbox,kz.bbox)>0.5: suppressed=True; break
            if not suppressed: kept.append(zone)
        return kept

    def _build_risk_map(self,zones,height,width):
        risk_map=np.zeros((height,width),dtype=np.float32)
        for zone in zones:
            if zone.mask is not None and zone.mask.shape==(height,width):
                risk_map[zone.mask]=np.maximum(risk_map[zone.mask],zone.severity)
            else:
                x1,y1,x2,y2=zone.bbox
                x1,y1=max(0,x1),max(0,y1); x2,y2=min(width,x2),min(height,y2)
                risk_map[y1:y2,x1:x2]=np.maximum(risk_map[y1:y2,x1:x2],zone.severity)
        if risk_map.max()>0:
            risk_map=cv2.GaussianBlur(risk_map,(51,51),sigmaX=15)
            risk_map/=risk_map.max()+1e-9
        return risk_map

    def _analyze_worker_proximity(self,zones,worker_boxes):
        for zone in zones:
            count=0
            for box in worker_boxes:
                wx1,wy1,wx2,wy2=box; wcx,wcy=(wx1+wx2)/2,(wy1+wy2)/2
                zx1,zy1,zx2,zy2=zone.bbox; pt=self.proximity_threshold
                if zx1-pt<=wcx<=zx2+pt and zy1-pt<=wcy<=zy2+pt: count+=1
            zone.worker_proximity_count=count
            if count>0: zone.severity=min(1.0,zone.severity*(1.0+0.2*count))
        return zones

    def _compute_overall_risk(self,zones,risk_map):
        if not zones: return 0.0
        total_weighted=sum(z.severity*z.area_pixels for z in zones)
        total_area=sum(z.area_pixels for z in zones)
        base=total_weighted/total_area if total_area>0 else 0.0
        penalty=sum(0.1*z.worker_proximity_count for z in zones)
        return min(1.0,base+penalty)

    def _risk_score_to_level(self,score):
        if score>=0.75: return "critical"
        elif score>=0.50: return "high"
        elif score>=0.25: return "medium"
        return "low"

    def _generate_alerts(self,zones,overall_risk):
        alerts=[]
        shafts=[z for z in zones if z.hazard_type==HazardType.OPEN_SHAFT]
        if shafts: alerts.append(f"CRITICAL: {len(shafts)} open shaft(s) detected")
        workers_at_risk=sum(z.worker_proximity_count for z in zones)
        if workers_at_risk: alerts.append(f"{workers_at_risk} worker(s) near hazard zones")
        if overall_risk>=0.75: alerts.append("Overall site risk: CRITICAL")
        return alerts

    @staticmethod
    def _bbox_iou(box_a,box_b):
        ax1,ay1,ax2,ay2=box_a; bx1,by1,bx2,by2=box_b
        ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
        if ix2<=ix1 or iy2<=iy1: return 0.0
        inter=(ix2-ix1)*(iy2-iy1); area_a=(ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by1)
        return inter/(area_a+area_b-inter+1e-9)

def colorize_risk_map(risk_map,colormap=cv2.COLORMAP_JET,alpha=0.6,base_image=None):
    risk_uint8=(risk_map*255).astype(np.uint8)
    colored=cv2.applyColorMap(risk_uint8,colormap)
    if base_image is not None:
        base_bgr=cv2.cvtColor(base_image,cv2.COLOR_RGB2BGR) if base_image.ndim==3 else base_image
        return cv2.addWeighted(base_bgr,1-alpha,colored,alpha,0)
    return colored
