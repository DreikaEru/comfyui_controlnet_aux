"""
DWPose Extended - Функции и классы
Этот файл содержит всю логику обработки
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import cv2
import torch

from .dwpose_3dData import (
    BONE_DEFINITIONS,
    BONE_COLORS,
    BoneCategory,
    FaceDirection,
    LimbPartType,
    GeometricPrimitive,
    LIMB_CHAINS,
    FINGER_NAMES,
    TOE_NAMES,
    PHALANX_COUNT,
    TOE_PHALANX_COUNT,
    COCO_TO_SKELETON,
    HAND_KEYPOINT_MAP,
    OPENPOSE_FALLBACK,
    UI_GROUPS,
    HEAD_BODY_PROPORTIONS,
    RefinementConfig,
)

# =============================================================================
# DEBUG
# =============================================================================

DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DWPose]", *args, **kwargs)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HeadProportions:
    head_center: Tuple[float, float]
    head_size: float
    head_width: float
    head_height: float
    
    def get_expected_size(self, bone_name: str) -> Tuple[float, float]:
        if bone_name not in HEAD_BODY_PROPORTIONS:
            return (0.01, 1.0)
        prop = HEAD_BODY_PROPORTIONS[bone_name]
        min_size = self.head_size * prop.size_relative_to_head[0]
        max_size = self.head_size * prop.size_relative_to_head[1]
        return (min_size, max_size)


@dataclass
class Contour:
    idx: int
    points: np.ndarray
    center: Tuple[float, float]
    size: Tuple[float, float]
    angle: float
    area: float
    depth: float
    depth_gradient: float
    cv_ellipse: Optional[Tuple] = None
    assigned_bone: Optional[str] = None
    search_score: float = 0.0


@dataclass
class BoneResult:
    bone_name: str
    position: Tuple[float, float]
    confidence: float
    depth: float
    contour: Optional[Contour]


@dataclass
class FaceAnalysis:
    center: Tuple[float, float]
    direction: FaceDirection
    confidence: float
    size: float
    nose_pos: Optional[Tuple[float, float]] = None
    eye_left: Optional[Tuple[float, float]] = None
    eye_right: Optional[Tuple[float, float]] = None
    chin: Optional[Tuple[float, float]] = None
    
    def get_neck_search_direction(self) -> Tuple[float, float]:
        return (0.0, 1.0)
    
    def get_shoulder_direction(self, side: str) -> Tuple[float, float]:
        if side == "L":
            return (-0.6, 0.4) if self.direction != FaceDirection.BACK else (-0.7, 0.3)
        else:
            return (0.6, 0.4) if self.direction != FaceDirection.BACK else (0.7, 0.3)


@dataclass
class PerspectiveAnalysis:
    tilt_angle: float
    near_body_parts: List[str]
    far_body_parts: List[str]
    scale_factors: Dict[str, float]
    is_full_body: bool
    
    def get_search_priority(self, bone_name: str) -> int:
        if bone_name in self.near_body_parts:
            return 90
        elif bone_name in self.far_body_parts:
            return 40
        else:
            return 60
    
    def get_expected_size_multiplier(self, bone_name: str) -> float:
        return self.scale_factors.get(bone_name, 1.0)


@dataclass
class DetectedPrimitive:
    primitive_type: GeometricPrimitive
    center: Tuple[float, float]
    size: Tuple[float, float]
    angle: float
    confidence: float
    contour: Optional[Contour]
    anatomical_label: Optional[str] = None


@dataclass
class BoneCandidate:
    bone_name: str
    position: Tuple[float, float]
    confidence: float
    depth: float
    source: str
    contour: Optional[Contour] = None
    primitive: Optional[DetectedPrimitive] = None
    
    def merge_with(self, other: 'BoneCandidate') -> 'BoneCandidate':
        total_conf = self.confidence + other.confidence
        new_x = (self.position[0] * self.confidence + other.position[0] * other.confidence) / total_conf
        new_y = (self.position[1] * self.confidence + other.position[1] * other.confidence) / total_conf
        new_depth = (self.depth * self.confidence + other.depth * other.confidence) / total_conf
        sources = [self.source, other.source]
        new_source = "+".join(sorted(set(sources)))
        new_conf = min(1.0, total_conf * 0.7)
        return BoneCandidate(
            bone_name=self.bone_name,
            position=(new_x, new_y),
            confidence=new_conf,
            depth=new_depth,
            source=new_source,
            contour=self.contour or other.contour,
            primitive=self.primitive or other.primitive
        )


# =============================================================================
# CONVEX NORMALS HELPER
# =============================================================================

class ConvexNormalsHelper:
    @staticmethod
    def compute_surface_normals(depth_map: np.ndarray) -> Optional[np.ndarray]:
        if depth_map is None or depth_map.size == 0:
            return None
        gx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        normals = np.zeros((*depth_map.shape, 3), dtype=np.float32)
        normals[:, :, 0] = -gx
        normals[:, :, 1] = -gy
        normals[:, :, 2] = 1.0
        norm = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True)) + 1e-8
        normals = normals / norm
        return normals
    
    @staticmethod
    def compute_convexity(depth_map: np.ndarray) -> Optional[np.ndarray]:
        if depth_map is None:
            return None
        laplacian = cv2.Laplacian(depth_map.astype(np.float32), cv2.CV_32F, ksize=5)
        if laplacian.max() > laplacian.min():
            laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
        return laplacian
    
    @staticmethod
    def visualize_normals(normals: np.ndarray) -> Optional[np.ndarray]:
        if normals is None:
            return None
        vis = ((normals + 1.0) * 127.5).astype(np.uint8)
        return vis


# =============================================================================
# DEPTH PROCESSOR
# =============================================================================

class DepthProcessor:
    def __init__(self, model_name: str = "midas_v21_small"):
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self.device = None
    
    def _get_best_device(self):
        if self.device is not None:
            return self.device
        try:
            import torch_directml
            self.device = torch_directml.device()
            debug_print(f"Using DirectML device")
            return self.device
        except ImportError:
            pass
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            debug_print(f"Using CUDA device")
            return self.device
        self.device = torch.device("cpu")
        debug_print(f"Using CPU device")
        return self.device
    
    def load_model(self):
        if self._loaded:
            return
        try:
            import sys
            import os
            controlnet_aux_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "..", 
                "custom_controlnet_aux"
            )
            if os.path.exists(controlnet_aux_path):
                sys.path.insert(0, controlnet_aux_path)
            try:
                from midas import MidasDetector
                self.model = MidasDetector.from_pretrained("lllyasviel/Annotators")
                debug_print(f"MiDaS loaded successfully")
            except ImportError:
                from custom_midas import MidasDetector
                self.model = MidasDetector.from_pretrained()
                debug_print(f"MiDaS loaded (alternative)")
            device = self._get_best_device()
            if hasattr(self.model, 'model') and device is not None:
                try:
                    self.model.model.to(device)
                except Exception:
                    pass
        except Exception as e:
            debug_print(f"Depth model load failed: {e}")
            self.model = None
        self._loaded = True
    
    def process(self, image: np.ndarray) -> Optional[np.ndarray]:
        if not self._loaded:
            self.load_model()
        if self.model is None:
            debug_print("Depth model not available")
            return None
        try:
            depth = self.model(image)
            if isinstance(depth, tuple):
                depth = depth[0]
            if depth.ndim == 3:
                depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
            depth = depth.astype(np.float32)
            if depth.max() > 1:
                depth = depth / 255.0
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth = (depth - d_min) / (d_max - d_min)
            return depth
        except Exception as e:
            debug_print(f"Depth processing error: {e}")
            return None
    
    def unload(self):
        if self.model is not None:
            del self.model
        self.model = None
        self._loaded = False


# =============================================================================
# EDGE PROCESSOR
# =============================================================================

class EdgeProcessor:
    def __init__(self, model_name: str = "canny"):
        self.model_name = model_name
        self.model = None
        self._loaded = False
    
    def load_model(self):
        if self._loaded:
            return
        if self.model_name != "canny":
            try:
                import sys
                import os
                controlnet_aux_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 
                    "..", 
                    "custom_controlnet_aux"
                )
                if os.path.exists(controlnet_aux_path):
                    sys.path.insert(0, controlnet_aux_path)
                if self.model_name == "hed":
                    try:
                        from hed import HEDdetector
                        self.model = HEDdetector.from_pretrained("lllyasviel/Annotators")
                        debug_print(f"HED loaded successfully")
                    except ImportError:
                        debug_print(f"HED not available, using Canny")
                        self.model_name = "canny"
            except Exception as e:
                debug_print(f"Edge model load failed: {e}, using Canny")
                self.model_name = "canny"
        self._loaded = True
    
    def process(self, image: np.ndarray, low: int = 30, high: int = 100) -> np.ndarray:
        if not self._loaded:
            self.load_model()
        if self.model_name == "canny" or self.model is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            return cv2.Canny(gray, low, high)
        try:
            edges = self.model(image)
            if edges.ndim == 3:
                edges = cv2.cvtColor(edges, cv2.COLOR_RGB2GRAY)
            return edges.astype(np.uint8)
        except Exception as e:
            debug_print(f"Edge processing error: {e}, using Canny")
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            return cv2.Canny(gray, low, high)
    
    def unload(self):
        if self.model is not None:
            del self.model
        self.model = None
        self._loaded = False


# =============================================================================
# CONTOUR FINDER
# =============================================================================

class ContourFinder:
    def __init__(self, config: RefinementConfig):
        self.config = config
    
    def find_contours(
        self,
        edge_map: np.ndarray,
        depth_map: Optional[np.ndarray],
        body_mask: Optional[np.ndarray],
        convexity_map: Optional[np.ndarray],
        image_size: Tuple[int, int]
    ) -> List[Contour]:
        h, w = image_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)
        contours_cv, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        body_depth_range = (0.0, 1.0)
        if depth_map is not None and body_mask is not None:
            body_pixels = depth_map[body_mask > 128]
            if len(body_pixels) > 0:
                d_min = np.percentile(body_pixels, 10)
                d_max = np.percentile(body_pixels, 90)
                body_depth_range = (float(d_min), float(d_max))
        
        contours = []
        for idx, cnt in enumerate(contours_cv):
            if len(cnt) < 5:
                continue
            area = cv2.contourArea(cnt)
            if area < self.config.min_contour_area:
                continue
            try:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), angle = rect
                cx_n, cy_n = cx / w, cy / h
                rw_n, rh_n = rw / w, rh / h
                
                depth_val = 0.5
                if depth_map is not None:
                    dh, dw = depth_map.shape[:2]
                    dx_map = max(0, min(dw-1, int(cx_n * dw)))
                    dy_map = max(0, min(dh-1, int(cy_n * dh)))
                    depth_val = float(depth_map[dy_map, dx_map])
                
                d_min, d_max = body_depth_range
                margin = (d_max - d_min) * 0.2 if d_max > d_min else 0.2
                if not (d_min - margin <= depth_val <= d_max + margin):
                    continue
                
                if body_mask is not None:
                    mh, mw = body_mask.shape[:2]
                    mx = max(0, min(mw-1, int(cx_n * mw)))
                    my = max(0, min(mh-1, int(cy_n * mh)))
                    radius = 5
                    region = body_mask[
                        max(0, my-radius):min(mh, my+radius+1),
                        max(0, mx-radius):min(mw, mx+radius+1)
                    ]
                    if region.size == 0 or (region > 128).sum() / region.size < 0.3:
                        continue
                
                cv_ellipse = None
                try:
                    cv_ellipse = cv2.fitEllipse(cnt)
                except Exception:
                    pass
                
                contours.append(Contour(
                    idx=idx,
                    points=cnt,
                    center=(cx_n, cy_n),
                    size=(rw_n, rh_n),
                    angle=angle,
                    area=area / (w * h),
                    depth=depth_val,
                    depth_gradient=0.0,
                    cv_ellipse=cv_ellipse
                ))
            except Exception:
                continue
        
        debug_print(f"Found {len(contours)} valid contours")
        return contours


# =============================================================================
# FACE DIRECTION ANALYZER
# =============================================================================

class FaceDirectionAnalyzer:
    @staticmethod
    def analyze(face_keypoints: List[Tuple[float, float, float]], head_pos: Tuple[float, float]) -> FaceAnalysis:
        if not face_keypoints or len(face_keypoints) < 5:
            return FaceAnalysis(
                center=head_pos,
                direction=FaceDirection.FRONT,
                confidence=0.3,
                size=0.12
            )
        all_pts = [fp for fp in face_keypoints if len(fp) >= 3 and fp[2] > 0.3]
        if all_pts:
            face_center = (
                sum(pt[0] for pt in all_pts) / len(all_pts),
                sum(pt[1] for pt in all_pts) / len(all_pts)
            )
            xs = [pt[0] for pt in all_pts]
            ys = [pt[1] for pt in all_pts]
            face_size = max(max(xs) - min(xs), max(ys) - min(ys))
        else:
            face_center = head_pos
            face_size = 0.12
        direction = FaceDirection.FRONT
        confidence = 0.7
        return FaceAnalysis(
            center=face_center,
            direction=direction,
            confidence=confidence,
            size=face_size
        )


# =============================================================================
# PERSPECTIVE ANALYZER
# =============================================================================

class PerspectiveAnalyzer:
    @staticmethod
    def analyze(
        depth_map: np.ndarray,
        body_mask: np.ndarray,
        head_pos: Tuple[float, float],
        dwpose_kp: Dict[str, Tuple],
        img_size: Tuple[int, int]
    ) -> PerspectiveAnalysis:
        if depth_map is None:
            return PerspectiveAnalysis(
                tilt_angle=0,
                near_body_parts=["Head", "Hand_L", "Hand_R"],
                far_body_parts=[],
                scale_factors={},
                is_full_body=False
            )
        h, w = img_size
        dh, dw = depth_map.shape[:2]
        depth_samples = {}
        hx = int(head_pos[0] * dw)
        hy = int(head_pos[1] * dh)
        depth_samples["Head"] = float(depth_map[
            max(0, min(dh-1, hy)),
            max(0, min(dw-1, hx))
        ])
        for bone_name, kp in dwpose_kp.items():
            if kp[2] > 0.3:
                bx = int(kp[0] * dw)
                by = int(kp[1] * dh)
                depth_samples[bone_name] = float(depth_map[
                    max(0, min(dh-1, by)),
                    max(0, min(dw-1, bx))
                ])
        if not depth_samples:
            return PerspectiveAnalysis(0, [], [], {}, False)
        sorted_parts = sorted(depth_samples.items(), key=lambda x: x[1], reverse=True)
        num_near = max(1, len(sorted_parts) // 3)
        near_parts = [name for name, _ in sorted_parts[:num_near]]
        num_far = max(1, len(sorted_parts) // 3)
        far_parts = [name for name, _ in sorted_parts[-num_far:]]
        is_full_body = any("Foot" in name or "Calf" in name or "Thigh" in name 
                          for name in depth_samples.keys())
        depth_range = sorted_parts[0][1] - sorted_parts[-1][1]
        scale_factors = {}
        if depth_range > 0.1:
            max_depth = sorted_parts[0][1]
            min_depth = sorted_parts[-1][1]
            for bone_name, depth in depth_samples.items():
                normalized_depth = (depth - min_depth) / depth_range
                scale_factors[bone_name] = 0.7 + normalized_depth * 0.6
        head_depth = depth_samples.get("Head", 0.5)
        legs_depth = depth_samples.get("Foot_L", depth_samples.get("Foot_R", 
                     depth_samples.get("Calf_L", depth_samples.get("Calf_R", head_depth))))
        depth_diff = legs_depth - head_depth
        tilt_angle = depth_diff * 45
        debug_print(f"Perspective: near={near_parts}, far={far_parts}, tilt={tilt_angle:.1f}°")
        return PerspectiveAnalysis(
            tilt_angle=tilt_angle,
            near_body_parts=near_parts,
            far_body_parts=far_parts,
            scale_factors=scale_factors,
            is_full_body=is_full_body
        )


# =============================================================================
# SILHOUETTE ANALYZER
# =============================================================================

class SilhouetteAnalyzer:
    @staticmethod
    def extract_silhouette(normals: np.ndarray, body_mask: np.ndarray, depth_map: np.ndarray) -> Optional[np.ndarray]:
        if body_mask is None:
            return None
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(body_mask, kernel, iterations=1)
        silhouette = cv2.subtract(body_mask, eroded)
        return silhouette
    
    @staticmethod
    def find_convex_regions(
        normals: np.ndarray,
        depth_map: np.ndarray,
        body_mask: np.ndarray,
        img_size: Tuple[int, int]
    ) -> List[Tuple[Tuple[float, float], float, str]]:
        if normals is None or body_mask is None:
            return []
        h, w = img_size
        z_normals = normals[:, :, 2]
        convex_mask = ((z_normals > 0.7) & (body_mask > 128)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        convex_mask = cv2.morphologyEx(convex_mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convex_mask, connectivity=8)
        regions = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:
                continue
            cx, cy = centroids[i]
            cx_n, cy_n = cx / w, cy / h
            region_w = stats[i, cv2.CC_STAT_WIDTH] / w
            region_h = stats[i, cv2.CC_STAT_HEIGHT] / h
            region_size = max(region_w, region_h)
            aspect = region_w / region_h if region_h > 0 else 1
            if 0.7 <= aspect <= 1.3 and region_size > 0.1:
                region_type = "head"
            elif region_size > 0.15:
                region_type = "torso"
            else:
                region_type = "limb"
            regions.append(((cx_n, cy_n), region_size, region_type))
        debug_print(f"Found {len(regions)} convex regions")
        return regions


# =============================================================================
# GEOMETRIC PRIMITIVE DETECTOR
# =============================================================================

class GeometricPrimitiveDetector:
    @staticmethod
    def detect_primitives(
        contours: List[Contour],
        convex_regions: List[Tuple],
        perspective: PerspectiveAnalysis,
        img_size: Tuple[int, int]
    ) -> List[DetectedPrimitive]:
        primitives = []
        for cnt in contours:
            primitive = GeometricPrimitiveDetector._analyze_contour_shape(cnt, perspective, img_size)
            if primitive:
                for region_center, region_size, region_type in convex_regions:
                    dist = math.sqrt(
                        (primitive.center[0] - region_center[0])**2 +
                        (primitive.center[1] - region_center[1])**2
                    )
                    if dist < 0.1:
                        primitive.confidence *= 1.2
                        primitive.confidence = min(1.0, primitive.confidence)
                primitives.append(primitive)
        debug_print(f"Detected {len(primitives)} geometric primitives")
        return primitives
    
    @staticmethod
    def _analyze_contour_shape(cnt: Contour, perspective: PerspectiveAnalysis, img_size: Tuple[int, int]) -> Optional[DetectedPrimitive]:
        if cnt.points is None or len(cnt.points) < 5:
            return None
        try:
            ellipse = cv2.fitEllipse(cnt.points)
            (cx, cy), (w_e, h_e), angle = ellipse
            h, w = img_size
            cx_n, cy_n = cx / w, cy / h
            w_e_n, h_e_n = w_e / w, h_e / h
            aspect = min(w_e, h_e) / max(w_e, h_e) if max(w_e, h_e) > 0 else 0
            contour_area = cnt.area
            ellipse_area = math.pi * (w_e / 2) * (h_e / 2) / (w * h)
            fit_quality = contour_area / ellipse_area if ellipse_area > 0 else 0
            if 0.8 <= aspect <= 1.0 and fit_quality > 0.8:
                if max(w_e_n, h_e_n) > 0.1:
                    return DetectedPrimitive(
                        GeometricPrimitive.ELLIPSE,
                        (cx_n, cy_n),
                        (w_e_n, h_e_n),
                        angle,
                        fit_quality * 0.9,
                        cnt,
                        "head_candidate"
                    )
                else:
                    return DetectedPrimitive(
                        GeometricPrimitive.CIRCLE,
                        (cx_n, cy_n),
                        (w_e_n, h_e_n),
                        angle,
                        fit_quality * 0.8,
                        cnt,
                        "joint_candidate"
                    )
            elif 0.3 <= aspect < 0.7 and fit_quality > 0.7:
                return DetectedPrimitive(
                    GeometricPrimitive.CONE,
                    (cx_n, cy_n),
                    (w_e_n, h_e_n),
                    angle,
                    fit_quality * 0.85,
                    cnt,
                    "limb_candidate"
                )
            elif 0.7 <= aspect < 0.9 and max(w_e_n, h_e_n) > 0.2:
                is_trapezoid = GeometricPrimitiveDetector._check_trapezoid(cnt.points)
                if is_trapezoid:
                    return DetectedPrimitive(
                        GeometricPrimitive.TRAPEZOID,
                        (cx_n, cy_n),
                        (w_e_n, h_e_n),
                        angle,
                        fit_quality * 0.9,
                        cnt,
                        "torso_candidate"
                    )
                else:
                    return DetectedPrimitive(
                        GeometricPrimitive.RECTANGLE,
                        (cx_n, cy_n),
                        (w_e_n, h_e_n),
                        angle,
                        fit_quality * 0.85,
                        cnt,
                        "torso_candidate"
                    )
        except Exception:
            pass
        return None
    
    @staticmethod
    def _check_trapezoid(contour_points: np.ndarray) -> bool:
        epsilon = 0.02 * cv2.arcLength(contour_points, True)
        approx = cv2.approxPolyDP(contour_points, epsilon, True)
        if 4 <= len(approx) <= 6:
            hull = cv2.convexHull(contour_points, returnPoints=False)
            try:
                defects = cv2.convexityDefects(contour_points, hull)
                if defects is not None:
                    significant = sum(1 for d in defects if d[0][3] > 2000)
                    return 2 <= significant <= 4
            except Exception:
                pass
        return False

# =============================================================================
# ПРОДОЛЖЕНИЕ dwpose_3dFunctions.py
# =============================================================================

# =============================================================================
# ANATOMICAL VALIDATOR
# =============================================================================

class AnatomicalValidator:
    MAX_BONE_DISTANCES = {
        ("Head", "Neck"): 1.2,
        ("Neck", "Clavicle_L"): 0.8,
        ("Neck", "Clavicle_R"): 0.8,
        ("Clavicle_L", "Shoulder_L"): 1.0,
        ("Clavicle_R", "Shoulder_R"): 1.0,
        ("Shoulder_L", "Forearm_L"): 1.5,
        ("Shoulder_R", "Forearm_R"): 1.5,
        ("Forearm_L", "Hand_L"): 1.5,
        ("Forearm_R", "Hand_R"): 1.5,
        ("Pelvis", "Thigh_L"): 0.8,
        ("Pelvis", "Thigh_R"): 0.8,
        ("Thigh_L", "Calf_L"): 1.5,
        ("Thigh_R", "Calf_R"): 1.5,
        ("Calf_L", "Foot_L"): 1.5,
        ("Calf_R", "Foot_R"): 1.5,
    }
    
    @staticmethod
    def validate_connection(
        bone1_name: str,
        bone1_pos: Tuple[float, float],
        bone2_name: str,
        bone2_pos: Tuple[float, float],
        head_size: float
    ) -> Tuple[bool, float]:
        dist = math.sqrt(
            (bone2_pos[0] - bone1_pos[0])**2 +
            (bone2_pos[1] - bone1_pos[1])**2
        )
        key = (bone1_name, bone2_name)
        max_dist = AnatomicalValidator.MAX_BONE_DISTANCES.get(key, 999.0) * head_size
        if dist > max_dist:
            return False, 0.0
        confidence = max(0.0, 1.0 - dist / max_dist)
        return True, confidence
    
    @staticmethod
    def validate_skeleton(
        bones: Dict[str, BoneResult],
        connections: List[Tuple[str, str]],
        head_size: float
    ) -> Dict[str, float]:
        scores = {}
        for bone1_name, bone2_name in connections:
            if bone1_name not in bones or bone2_name not in bones:
                continue
            bone1 = bones[bone1_name]
            bone2 = bones[bone2_name]
            valid, conf = AnatomicalValidator.validate_connection(
                bone1_name, bone1.position,
                bone2_name, bone2.position,
                head_size
            )
            if valid:
                scores[bone1_name] = max(scores.get(bone1_name, 0), conf)
                scores[bone2_name] = max(scores.get(bone2_name, 0), conf)
            else:
                scores[bone2_name] = min(scores.get(bone2_name, 1.0), 0.3)
        return scores


# =============================================================================
# MULTI-CHANNEL SKELETON BUILDER
# =============================================================================

class MultiChannelSkeletonBuilder:
    def __init__(self):
        self.perspective = None
        self.head_props = None
        self.face_analysis = None
        self.primitives = []
        self.convex_regions = []
        self.silhouette = None
    
    def build_skeleton(
        self,
        image: np.ndarray,
        dwpose_kp: Dict[str, Tuple],
        face_kp: List,
        contours: List[Contour],
        depth_map: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        body_mask: Optional[np.ndarray],
        enabled_bones: Set[str],
        img_size: Tuple[int, int]
    ) -> Tuple[Dict[str, BoneResult], Dict[str, Any]]:
        h, w = img_size
        
        # Анализ головы
        head_pos = None
        if "Head" in dwpose_kp and dwpose_kp["Head"][2] > 0.3:
            head_pos = (dwpose_kp["Head"][0], dwpose_kp["Head"][1])
        
        if not head_pos:
            debug_print("ERROR: No head position")
            return {}, {}
        
        self.perspective = PerspectiveAnalyzer.analyze(
            depth_map, body_mask, head_pos, dwpose_kp, img_size
        )
        
        self.face_analysis = FaceDirectionAnalyzer.analyze(face_kp, head_pos)
        
        self.head_props = HeadProportions(
            head_center=head_pos,
            head_size=self.face_analysis.size,
            head_width=self.face_analysis.size * 0.8,
            head_height=self.face_analysis.size * 1.2
        )
        
        # Силуэт и выпуклые области
        if normals is not None and body_mask is not None:
            self.silhouette = SilhouetteAnalyzer.extract_silhouette(
                normals, body_mask, depth_map
            )
            self.convex_regions = SilhouetteAnalyzer.find_convex_regions(
                normals, depth_map, body_mask, img_size
            )
        
        # Геометрические примитивы
        self.primitives = GeometricPrimitiveDetector.detect_primitives(
            contours, self.convex_regions, self.perspective, img_size
        )
        
        # Сбор кандидатов из всех каналов
        all_candidates = defaultdict(list)
        
        # Канал 1: DWPose
        for bone_name, kp in dwpose_kp.items():
            if kp[2] > 0.3 and bone_name in enabled_bones:
                depth = self._get_depth(kp[0], kp[1], depth_map, w, h)
                all_candidates[bone_name].append(BoneCandidate(
                    bone_name=bone_name,
                    position=(kp[0], kp[1]),
                    confidence=kp[2] * 0.9,
                    depth=depth,
                    source="dwpose"
                ))
        
        # Канал 2: Геометрические примитивы
        primitive_candidates = self._match_primitives_to_bones(
            self.primitives, enabled_bones, depth_map, w, h
        )
        for bone_name, candidate in primitive_candidates.items():
            all_candidates[bone_name].append(candidate)
        
        # Канал 3: Выпуклые области
        convex_candidates = self._match_convex_to_bones(
            self.convex_regions, enabled_bones, depth_map, w, h
        )
        for bone_name, candidate in convex_candidates.items():
            all_candidates[bone_name].append(candidate)
        
        # Канал 4: Анатомическая оценка
        estimated = self._estimate_missing_bones(
            all_candidates, enabled_bones, depth_map, w, h
        )
        for bone_name, candidate in estimated.items():
            all_candidates[bone_name].append(candidate)
        
        # Объединение кандидатов
        merged_candidates = {}
        for bone_name, candidates in all_candidates.items():
            if len(candidates) == 1:
                merged_candidates[bone_name] = candidates[0]
            else:
                merged = self._merge_candidates(candidates)
                merged_candidates[bone_name] = merged
        
        # Построение скелета с приоритетами
        bones = self._build_with_priorities(merged_candidates, enabled_bones)
        
        # Валидация
        connections = self._get_connections(bones, enabled_bones)
        validation_scores = AnatomicalValidator.validate_skeleton(
            bones, connections, self.head_props.head_size
        )
        
        for bone_name, score in validation_scores.items():
            if bone_name in bones:
                bones[bone_name].confidence *= (0.7 + score * 0.3)
        
        # Производные кости
        self._add_derived_bones(bones, enabled_bones, depth_map, w, h)
        
        metadata = {
            "perspective": self.perspective,
            "primitives": self.primitives,
            "convex_regions": self.convex_regions,
            "silhouette": self.silhouette,
            "validation_scores": validation_scores,
            "all_candidates": all_candidates,
        }
        
        return bones, metadata
    
    def _match_primitives_to_bones(self, primitives, enabled_bones, depth_map, w, h):
        candidates = {}
        for prim in primitives:
            if prim.primitive_type == GeometricPrimitive.ELLIPSE:
                if max(prim.size) > 0.1 and "Head" in enabled_bones:
                    candidates["Head"] = BoneCandidate(
                        bone_name="Head",
                        position=prim.center,
                        confidence=prim.confidence * 0.85,
                        depth=self._get_depth(prim.center[0], prim.center[1], depth_map, w, h),
                        source="primitive",
                        primitive=prim
                    )
            elif prim.primitive_type in [GeometricPrimitive.TRAPEZOID, GeometricPrimitive.RECTANGLE]:
                if "Neck" in enabled_bones and "Neck" not in candidates:
                    neck_y = prim.center[1] - prim.size[1] * 0.4
                    candidates["Neck"] = BoneCandidate(
                        bone_name="Neck",
                        position=(prim.center[0], neck_y),
                        confidence=prim.confidence * 0.7,
                        depth=self._get_depth(prim.center[0], neck_y, depth_map, w, h),
                        source="primitive",
                        primitive=prim
                    )
            elif prim.primitive_type == GeometricPrimitive.CONE:
                size = max(prim.size)
                y_pos = prim.center[1]
                if y_pos > 0.5 and size > 0.08:
                    side = "L" if prim.center[0] < 0.5 else "R"
                    bone_name = f"Thigh_{side}" if y_pos < 0.7 else f"Calf_{side}"
                    if bone_name in enabled_bones and bone_name not in candidates:
                        candidates[bone_name] = BoneCandidate(
                            bone_name=bone_name,
                            position=prim.center,
                            confidence=prim.confidence * 0.75,
                            depth=self._get_depth(prim.center[0], prim.center[1], depth_map, w, h),
                            source="primitive",
                            primitive=prim
                        )
                elif y_pos < 0.6 and size > 0.06:
                    side = "L" if prim.center[0] < 0.5 else "R"
                    bone_name = f"Shoulder_{side}" if y_pos < 0.4 else f"Forearm_{side}"
                    if bone_name in enabled_bones and bone_name not in candidates:
                        candidates[bone_name] = BoneCandidate(
                            bone_name=bone_name,
                            position=prim.center,
                            confidence=prim.confidence * 0.75,
                            depth=self._get_depth(prim.center[0], prim.center[1], depth_map, w, h),
                            source="primitive",
                            primitive=prim
                        )
            elif prim.primitive_type == GeometricPrimitive.CIRCLE:
                if prim.center[1] > 0.8:
                    side = "L" if prim.center[0] < 0.5 else "R"
                    bone_name = f"Foot_{side}"
                    if bone_name in enabled_bones and bone_name not in candidates:
                        candidates[bone_name] = BoneCandidate(
                            bone_name=bone_name,
                            position=prim.center,
                            confidence=prim.confidence * 0.7,
                            depth=self._get_depth(prim.center[0], prim.center[1], depth_map, w, h),
                            source="primitive",
                            primitive=prim
                        )
        return candidates
    
    def _match_convex_to_bones(self, convex_regions, enabled_bones, depth_map, w, h):
        candidates = {}
        for center, size, region_type in convex_regions:
            if region_type == "head" and "Head" in enabled_bones and "Head" not in candidates:
                candidates["Head"] = BoneCandidate(
                    bone_name="Head",
                    position=center,
                    confidence=0.75,
                    depth=self._get_depth(center[0], center[1], depth_map, w, h),
                    source="silhouette"
                )
            elif region_type == "torso":
                if "Neck" in enabled_bones and "Neck" not in candidates:
                    neck_y = center[1] - size * 0.3
                    candidates["Neck"] = BoneCandidate(
                        bone_name="Neck",
                        position=(center[0], neck_y),
                        confidence=0.65,
                        depth=self._get_depth(center[0], neck_y, depth_map, w, h),
                        source="silhouette"
                    )
        return candidates
    
    def _estimate_missing_bones(self, existing_candidates, enabled_bones, depth_map, w, h):
        estimated = {}
        if "Head" in existing_candidates and "Neck" not in existing_candidates and "Neck" in enabled_bones:
            head_pos = existing_candidates["Head"][0].position
            neck_pos = (head_pos[0], head_pos[1] + self.head_props.head_size * 0.9)
            estimated["Neck"] = BoneCandidate(
                bone_name="Neck",
                position=neck_pos,
                confidence=0.5,
                depth=self._get_depth(neck_pos[0], neck_pos[1], depth_map, w, h),
                source="estimated"
            )
        if "Neck" in existing_candidates:
            neck_pos = existing_candidates["Neck"][0].position
            for side in ["L", "R"]:
                shoulder_key = f"Shoulder_{side}"
                if shoulder_key not in existing_candidates and shoulder_key in enabled_bones:
                    offset_x = -0.12 if side == "L" else 0.12
                    shoulder_pos = (neck_pos[0] + offset_x, neck_pos[1] + 0.08)
                    estimated[shoulder_key] = BoneCandidate(
                        bone_name=shoulder_key,
                        position=shoulder_pos,
                        confidence=0.4,
                        depth=self._get_depth(shoulder_pos[0], shoulder_pos[1], depth_map, w, h),
                        source="estimated"
                    )
        return estimated
    
    def _merge_candidates(self, candidates):
        if len(candidates) == 1:
            return candidates[0]
        groups = []
        for candidate in candidates:
            added = False
            for group in groups:
                for member in group:
                    dist = math.sqrt(
                        (candidate.position[0] - member.position[0])**2 +
                        (candidate.position[1] - member.position[1])**2
                    )
                    if dist < 0.05:
                        group.append(candidate)
                        added = True
                        break
                if added:
                    break
            if not added:
                groups.append([candidate])
        best_group = max(groups, key=lambda g: sum(c.confidence for c in g))
        result = best_group[0]
        for candidate in best_group[1:]:
            result = result.merge_with(candidate)
        return result
    
    def _build_with_priorities(self, candidates, enabled_bones):
        bones = {}
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: self.perspective.get_search_priority(x[0]),
            reverse=True
        )
        for bone_name, candidate in sorted_candidates:
            if bone_name in enabled_bones:
                bones[bone_name] = BoneResult(
                    bone_name=bone_name,
                    position=candidate.position,
                    confidence=candidate.confidence,
                    depth=candidate.depth,
                    contour=candidate.contour
                )
        return bones
    
    def _add_derived_bones(self, bones, enabled_bones, depth_map, w, h):
        # Spine
        neck = bones.get("Neck")
        pelvis = bones.get("Pelvis")
        if not pelvis and ("Thigh_L" in bones or "Thigh_R" in bones):
            thighs = [bones.get("Thigh_L"), bones.get("Thigh_R")]
            thighs = [t for t in thighs if t]
            if thighs:
                pelvis_x = sum(t.position[0] for t in thighs) / len(thighs)
                pelvis_y = sum(t.position[1] for t in thighs) / len(thighs) - 0.02
                bones["Pelvis"] = BoneResult(
                    bone_name="Pelvis",
                    position=(pelvis_x, pelvis_y),
                    confidence=0.6,
                    depth=self._get_depth(pelvis_x, pelvis_y, depth_map, w, h),
                    contour=None
                )
                pelvis = bones["Pelvis"]
        if neck and pelvis:
            for i, key in enumerate(["Spine_3", "Spine_2", "Spine_1"], 1):
                if key not in bones and key in enabled_bones:
                    t = i / 4.0
                    sx = pelvis.position[0] + (neck.position[0] - pelvis.position[0]) * t
                    sy = pelvis.position[1] + (neck.position[1] - pelvis.position[1]) * t
                    bones[key] = BoneResult(
                        bone_name=key,
                        position=(sx, sy),
                        confidence=min(neck.confidence, pelvis.confidence) * 0.8,
                        depth=self._get_depth(sx, sy, depth_map, w, h),
                        contour=None
                    )
        if pelvis and "Root" not in bones and "Root" in enabled_bones:
            bones["Root"] = BoneResult(
                bone_name="Root",
                position=pelvis.position,
                confidence=pelvis.confidence * 0.9,
                depth=pelvis.depth,
                contour=None
            )
        if neck:
            for side in ["L", "R"]:
                ck = f"Clavicle_{side}"
                sk = f"Shoulder_{side}"
                if ck not in bones and ck in enabled_bones and sk in bones:
                    shoulder = bones[sk]
                    cx = neck.position[0] * 0.35 + shoulder.position[0] * 0.65
                    cy = neck.position[1] * 0.35 + shoulder.position[1] * 0.65
                    bones[ck] = BoneResult(
                        bone_name=ck,
                        position=(cx, cy),
                        confidence=min(neck.confidence, shoulder.confidence) * 0.85,
                        depth=self._get_depth(cx, cy, depth_map, w, h),
                        contour=None
                    )
    
    def _get_connections(self, bones, enabled_bones):
        connections = []
        for bone_name in bones.keys():
            bone_def = BONE_DEFINITIONS.get(bone_name)
            if bone_def and bone_def.parent:
                if bone_def.parent in bones:
                    connections.append((bone_def.parent, bone_name))
        if "Pelvis" not in bones:
            if "Neck" in bones:
                for side in ["L", "R"]:
                    thigh = f"Thigh_{side}"
                    if thigh in bones:
                        connections.append(("Neck", thigh))
        return connections
    
    def _get_depth(self, x, y, dm, w, h):
        if dm is None:
            return 0.5
        dh, dw = dm.shape[:2]
        return float(dm[max(0, min(dh-1, int(y*dh))), max(0, min(dw-1, int(x*dw)))])


# =============================================================================
# FINGER GENERATOR
# =============================================================================

class FingerGenerator:
    @staticmethod
    def generate_from_hand(
        hand_pos: Tuple[float, float, float, float],
        forearm_pos: Optional[Tuple[float, float, float, float]],
        side: str,
        enabled_fingers: Set[str]
    ) -> Dict[str, Tuple[float, float, float, float]]:
        result = {}
        hx, hy, conf, hz = hand_pos
        if forearm_pos:
            dx = hx - forearm_pos[0]
            dy = hy - forearm_pos[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0.01:
                dx, dy = dx/length, dy/length
            else:
                dx, dy = 0, 0.1
        else:
            dx, dy = 0, 0.1
        px, py = -dy, dx
        phalanx_len = 0.02
        finger_angles = {
            "Thumb": -0.6 if side == "R" else 0.6,
            "Index": -0.3 if side == "R" else 0.3,
            "Middle": 0,
            "Ring": 0.3 if side == "R" else -0.3,
            "Pinky": 0.6 if side == "R" else -0.6,
        }
        for finger in FINGER_NAMES:
            angle_offset = finger_angles.get(finger, 0)
            fx = dx + px * angle_offset
            fy = dy + py * angle_offset
            flen = math.sqrt(fx*fx + fy*fy)
            if flen > 0:
                fx, fy = fx/flen, fy/flen
            curr_x, curr_y = hx, hy
            for phalanx in range(1, PHALANX_COUNT + 1):
                bone_name = f"{finger}_{phalanx}_{side}"
                if bone_name not in enabled_fingers:
                    continue
                curr_x += fx * phalanx_len
                curr_y += fy * phalanx_len
                result[bone_name] = (
                    max(0, min(1, curr_x)),
                    max(0, min(1, curr_y)),
                    conf * (0.8 ** phalanx),
                    hz
                )
        return result
    
    @staticmethod
    def generate_toes(
        foot_pos: Tuple[float, float, float, float],
        side: str,
        enabled_toes: Set[