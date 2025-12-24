"""
DWPose Extended Preprocessor для ComfyUI
Двунаправленный поиск с анализом наклона по глубине и трапециевидным торсом
"""

from __future__ import annotations
import os
import json
import math
from typing import Dict, List, Optional, Tuple, Set
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import cv2
import torch
import comfy.model_management as model_management

from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT

from .dwpose_3dData import (
    DWPOSE_MODEL_NAME,
    MODEL_REPOS,
    GPU_PROVIDERS,
    DEPTH_MODELS,
    EDGE_MODELS,
    BONE_DEFINITIONS,
    UI_GROUPS,
    BoneCategory,
    COCO_TO_SKELETON,
    HAND_KEYPOINT_MAP,
    BONE_COLORS,
    OPENPOSE_FALLBACK,
    RefinementConfig,
    FINGER_NAMES,
    TOE_NAMES,
    PHALANX_COUNT,
    TOE_PHALANX_COUNT,
    LIMB_CHAINS,
    HEAD_BODY_PROPORTIONS,
)

# =============================================================================
# DEBUG
# =============================================================================

DEBUG = True


def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DWPose]", *args, **kwargs)


# =============================================================================
# GLOBAL INITIALIZATION
# =============================================================================

_USE_CPU_FALLBACK = False


def _set_cpu_fallback(value: bool = True):
    global _USE_CPU_FALLBACK
    _USE_CPU_FALLBACK = value


def _get_device():
    if _USE_CPU_FALLBACK:
        return torch.device("cpu")
    try:
        return model_management.get_torch_device()
    except Exception:
        _set_cpu_fallback(True)
        return torch.device("cpu")


def _init_onnx():
    global _USE_CPU_FALLBACK
    if os.environ.get("DWPOSE_ONNXRT_CHECKED"):
        return
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers() or []
        debug_print(f"Available ONNX providers: {providers}")
        
        # Приоритет: DirectML (AMD/Intel) > CUDA > CPU
        priority = ["DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider"]
        selected = [p for p in priority if p in providers]
        if "CPUExecutionProvider" not in selected:
            selected.append("CPUExecutionProvider")
        
        if selected[0] != "CPUExecutionProvider":
            os.environ.setdefault("AUX_ORT_PROVIDERS", ",".join(selected))
            debug_print(f"Using ONNX provider: {selected[0]}")
        else:
            _USE_CPU_FALLBACK = True
            
    except Exception as e:
        debug_print(f"ONNX init error: {e}")
        _USE_CPU_FALLBACK = True
    
    os.environ["DWPOSE_ONNXRT_CHECKED"] = "1"


_init_onnx()


# =============================================================================
# PERSPECTIVE ANALYZER (анализ перспективы для приоритетов поиска)
# =============================================================================

@dataclass
class PerspectiveAnalysis:
    """Анализ перспективы для определения приоритетов поиска"""
    tilt_angle: float  # угол наклона
    near_body_parts: List[str]  # части тела ближе к камере (крупнее, приоритет)
    far_body_parts: List[str]   # части тела дальше (мельче, второстепенно)
    scale_factors: Dict[str, float]  # коэффициенты масштаба для каждой части
    is_full_body: bool  # весь скелет виден
    
    def get_search_priority(self, bone_name: str) -> int:
        """Приоритет поиска (0-100, выше = раньше искать)"""
        if bone_name in self.near_body_parts:
            return 90
        elif bone_name in self.far_body_parts:
            return 40
        else:
            return 60
    
    def get_expected_size_multiplier(self, bone_name: str) -> float:
        """Множитель размера с учётом перспективы"""
        return self.scale_factors.get(bone_name, 1.0)


class PerspectiveAnalyzer:
    """
    Анализирует перспективу и определяет:
    1. Что ближе к камере (крупнее, искать в первую очередь)
    2. Что дальше (мельче, искать во вторую очередь)
    3. Коэффициенты масштаба для каждой части тела
    """
    
    @staticmethod
    def analyze(
        depth_map: np.ndarray,
        body_mask: np.ndarray,
        head_pos: Tuple[float, float],
        dwpose_kp: Dict[str, Tuple],
        img_size: Tuple[int, int]
    ) -> PerspectiveAnalysis:
        """Анализ перспективы"""
        
        if depth_map is None:
            # Без глубины - предполагаем фронтальный вид без перспективы
            return PerspectiveAnalysis(
                tilt_angle=0,
                near_body_parts=["Head", "Hand_L", "Hand_R"],
                far_body_parts=[],
                scale_factors={},
                is_full_body=False
            )
        
        h, w = img_size
        dh, dw = depth_map.shape[:2]
        
        # Собираем глубины доступных частей тела
        depth_samples = {}
        
        # Голова
        hx = int(head_pos[0] * dw)
        hy = int(head_pos[1] * dh)
        depth_samples["Head"] = float(depth_map[
            max(0, min(dh-1, hy)),
            max(0, min(dw-1, hx))
        ])
        
        # Другие части из DWPose
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
        
        # Сортируем по глубине
        sorted_parts = sorted(depth_samples.items(), key=lambda x: x[1], reverse=True)
        
        # Ближние (светлее) - первые 1/3
        num_near = max(1, len(sorted_parts) // 3)
        near_parts = [name for name, _ in sorted_parts[:num_near]]
        
        # Дальние (темнее) - последние 1/3
        num_far = max(1, len(sorted_parts) // 3)
        far_parts = [name for name, _ in sorted_parts[-num_far:]]
        
        # Проверка full body (есть ли ноги)
        is_full_body = any("Foot" in name or "Calf" in name or "Thigh" in name 
                          for name in depth_samples.keys())
        
        # Вычисляем коэффициенты масштаба
        # Ближние части кажутся крупнее
        depth_range = sorted_parts[0][1] - sorted_parts[-1][1]
        scale_factors = {}
        
        if depth_range > 0.1:  # Значительная разница глубины
            max_depth = sorted_parts[0][1]
            min_depth = sorted_parts[-1][1]
            
            for bone_name, depth in depth_samples.items():
                # Перспектива: ближе = крупнее
                # Нормализуем в диапазон [0.7, 1.3]
                normalized_depth = (depth - min_depth) / depth_range
                scale_factors[bone_name] = 0.7 + normalized_depth * 0.6
        
        # Угол наклона (упрощённо)
        head_depth = depth_samples.get("Head", 0.5)
        legs_depth = depth_samples.get("Foot_L", depth_samples.get("Foot_R", 
                     depth_samples.get("Calf_L", depth_samples.get("Calf_R", head_depth))))
        
        depth_diff = legs_depth - head_depth
        tilt_angle = depth_diff * 45  # Грубая оценка
        
        debug_print(f"Perspective analysis:")
        debug_print(f"  Near parts (priority): {near_parts}")
        debug_print(f"  Far parts (secondary): {far_parts}")
        debug_print(f"  Tilt angle: {tilt_angle:.1f}°")
        debug_print(f"  Full body: {is_full_body}")
        
        return PerspectiveAnalysis(
            tilt_angle=tilt_angle,
            near_body_parts=near_parts,
            far_body_parts=far_parts,
            scale_factors=scale_factors,
            is_full_body=is_full_body
        )


# =============================================================================
# SILHOUETTE ANALYZER (анализ силуэта по нормалям)
# =============================================================================

class SilhouetteAnalyzer:
    """
    Анализ силуэта тела по нормалям и маске
    Находит выпуклые области (голова, конечности, торс)
    """
    
    @staticmethod
    def extract_silhouette(
        normals: np.ndarray,
        body_mask: np.ndarray,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """Извлекает силуэт тела"""
        
        if body_mask is None:
            return None
        
        # Силуэт = граница маски
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
        """
        Находит выпуклые области (конечности)
        Возвращает: [(центр, размер, тип), ...]
        тип: "head", "limb", "torso"
        """
        
        if normals is None or body_mask is None:
            return []
        
        h, w = img_size
        
        # Z-компонента нормалей (направление к камере)
        # Высокие значения = поверхность смотрит на камеру (выпуклость)
        z_normals = normals[:, :, 2]
        
        # Порог для выпуклых областей
        convex_mask = ((z_normals > 0.7) & (body_mask > 128)).astype(np.uint8) * 255
        
        # Морфология для объединения близких областей
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        convex_mask = cv2.morphologyEx(convex_mask, cv2.MORPH_CLOSE, kernel)
        
        # Находим связные компоненты
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convex_mask, connectivity=8)
        
        regions = []
        
        for i in range(1, num_labels):  # Пропускаем фон (0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Фильтруем слишком мелкие
            if area < 100:
                continue
            
            cx, cy = centroids[i]
            cx_n, cy_n = cx / w, cy / h
            
            # Размер региона
            region_w = stats[i, cv2.CC_STAT_WIDTH] / w
            region_h = stats[i, cv2.CC_STAT_HEIGHT] / h
            region_size = max(region_w, region_h)
            
            # Классификация по форме
            aspect = region_w / region_h if region_h > 0 else 1
            
            if 0.7 <= aspect <= 1.3 and region_size > 0.1:
                region_type = "head"  # Круглая и крупная
            elif region_size > 0.15:
                region_type = "torso"  # Крупная
            else:
                region_type = "limb"  # Вытянутая
            
            regions.append(((cx_n, cy_n), region_size, region_type))
        
        debug_print(f"Found {len(regions)} convex regions: {[r[2] for r in regions]}")
        return regions


# =============================================================================
# GEOMETRIC PRIMITIVE DETECTOR (детектор геометрических примитивов)
# =============================================================================

class GeometricPrimitive(Enum):
    """Типы геометрических примитивов"""
    ELLIPSE = "ellipse"      # Голова, суставы
    TRAPEZOID = "trapezoid"  # Торс (front/back)
    RECTANGLE = "rectangle"  # Торс (side), конечности
    CONE = "cone"            # Бедро, голень, плечо
    CIRCLE = "circle"        # Пятка, колено
    COMPOSITE = "composite"  # Кисть, стопа (составная форма)


@dataclass
class DetectedPrimitive:
    """Обнаруженный геометрический примитив"""
    primitive_type: GeometricPrimitive
    center: Tuple[float, float]
    size: Tuple[float, float]  # width, height
    angle: float
    confidence: float
    contour: Optional['Contour']
    anatomical_label: Optional[str] = None  # "head", "torso", "thigh_L", etc.


class GeometricPrimitiveDetector:
    """
    Детектор геометрических примитивов в контурах
    Сопоставляет контуры с анатомическими формами
    """
    
    @staticmethod
    def detect_primitives(
        contours: List['Contour'],
        convex_regions: List[Tuple],
        perspective: PerspectiveAnalysis,
        img_size: Tuple[int, int]
    ) -> List[DetectedPrimitive]:
        """Детектирует геометрические примитивы"""
        
        primitives = []
        
        for cnt in contours:
            # Анализ формы контура
            primitive = GeometricPrimitiveDetector._analyze_contour_shape(cnt, perspective, img_size)
            
            if primitive:
                # Сопоставление с выпуклыми областями
                for region_center, region_size, region_type in convex_regions:
                    dist = math.sqrt(
                        (primitive.center[0] - region_center[0])**2 +
                        (primitive.center[1] - region_center[1])**2
                    )
                    
                    # Если близко к выпуклой области - бонус к уверенности
                    if dist < 0.1:
                        primitive.confidence *= 1.2
                        primitive.confidence = min(1.0, primitive.confidence)
                
                primitives.append(primitive)
        
        debug_print(f"Detected {len(primitives)} geometric primitives")
        return primitives
    
    @staticmethod
    def _analyze_contour_shape(
        cnt: 'Contour',
        perspective: PerspectiveAnalysis,
        img_size: Tuple[int, int]
    ) -> Optional[DetectedPrimitive]:
        """Анализирует форму контура и определяет примитив"""
        
        if cnt.points is None or len(cnt.points) < 5:
            return None
        
        # Эллипс
        try:
            ellipse = cv2.fitEllipse(cnt.points)
            (cx, cy), (w_e, h_e), angle = ellipse
            
            h, w = img_size
            cx_n, cy_n = cx / w, cy / h
            w_e_n, h_e_n = w_e / w, h_e / h
            
            # Соотношение сторон
            aspect = min(w_e, h_e) / max(w_e, h_e) if max(w_e, h_e) > 0 else 0
            
            # Насколько хорошо подходит эллипс
            contour_area = cnt.area
            ellipse_area = math.pi * (w_e / 2) * (h_e / 2) / (w * h)
            fit_quality = contour_area / ellipse_area if ellipse_area > 0 else 0
            
            # Классификация
            if 0.8 <= aspect <= 1.0 and fit_quality > 0.8:
                # Круглый - голова или сустав
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
                # Вытянутый - конечность или конус
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
                # Крупный овал - торс
                # Проверяем на трапецию
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
        """Проверяет, похож ли контур на трапецию"""
        
        # Упрощаем контур
        epsilon = 0.02 * cv2.arcLength(contour_points, True)
        approx = cv2.approxPolyDP(contour_points, epsilon, True)
        
        # Трапеция ≈ 4 угла
        if 4 <= len(approx) <= 6:
            # Проверяем вогнутости (изгибы над плечами)
            hull = cv2.convexHull(contour_points, returnPoints=False)
            try:
                defects = cv2.convexityDefects(contour_points, hull)
                if defects is not None:
                    # 2-3 значительных вогнутости = трапеция торса
                    significant = sum(1 for d in defects if d[0][3] > 2000)
                    return 2 <= significant <= 4
            except Exception:
                pass
        
        return False


# =============================================================================
# ANATOMICAL VALIDATOR (валидатор анатомических соединений)
# =============================================================================

class AnatomicalValidator:
    """
    Проверяет анатомическую валидность соединений костей
    """
    
    # Максимальные расстояния между соединёнными костями (в долях размера головы)
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
    
    # Ожидаемые углы между костями (min, max в градусах)
    EXPECTED_ANGLES = {
        ("Neck", "Clavicle_L", "Shoulder_L"): (30, 150),
        ("Neck", "Clavicle_R", "Shoulder_R"): (30, 150),
        ("Shoulder_L", "Forearm_L", "Hand_L"): (0, 180),
        ("Shoulder_R", "Forearm_R", "Hand_R"): (0, 180),
        ("Thigh_L", "Calf_L", "Foot_L"): (90, 180),
        ("Thigh_R", "Calf_R", "Foot_R"): (90, 180),
    }
    
    @staticmethod
    def validate_connection(
        bone1_name: str,
        bone1_pos: Tuple[float, float],
        bone2_name: str,
        bone2_pos: Tuple[float, float],
        head_size: float
    ) -> Tuple[bool, float]:
        """
        Проверяет валидность соединения двух костей
        Возвращает: (валидно, уверенность)
        """
        
        # Расстояние между костями
        dist = math.sqrt(
            (bone2_pos[0] - bone1_pos[0])**2 +
            (bone2_pos[1] - bone1_pos[1])**2
        )
        
        # Проверяем максимальное расстояние
        key = (bone1_name, bone2_name)
        max_dist = AnatomicalValidator.MAX_BONE_DISTANCES.get(key, 999.0) * head_size
        
        if dist > max_dist:
            return False, 0.0
        
        # Уверенность зависит от расстояния
        # Ближе = лучше
        confidence = max(0.0, 1.0 - dist / max_dist)
        
        return True, confidence
    
    @staticmethod
    def validate_angle(
        bone1_name: str,
        bone1_pos: Tuple[float, float],
        bone2_name: str,
        bone2_pos: Tuple[float, float],
        bone3_name: str,
        bone3_pos: Tuple[float, float]
    ) -> Tuple[bool, float]:
        """
        Проверяет угол между тремя костями (bone2 = вершина угла)
        """
        
        # Векторы
        v1 = (bone1_pos[0] - bone2_pos[0], bone1_pos[1] - bone2_pos[1])
        v2 = (bone3_pos[0] - bone2_pos[0], bone3_pos[1] - bone2_pos[1])
        
        # Длины
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 < 0.001 or len2 < 0.001:
            return True, 0.5  # Не можем проверить
        
        # Угол
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cos_angle = dot / (len1 * len2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))
        
        # Проверяем диапазон
        key = (bone1_name, bone2_name, bone3_name)
        expected_range = AnatomicalValidator.EXPECTED_ANGLES.get(key, (0, 180))
        
        if expected_range[0] <= angle_deg <= expected_range[1]:
            # Чем ближе к центру диапазона, тем лучше
            center = (expected_range[0] + expected_range[1]) / 2
            deviation = abs(angle_deg - center) / ((expected_range[1] - expected_range[0]) / 2)
            confidence = max(0.0, 1.0 - deviation * 0.5)
            return True, confidence
        else:
            return False, 0.0
    
    @staticmethod
    def validate_skeleton(
        bones: Dict[str, 'BoneResult'],
        connections: List[Tuple[str, str]],
        head_size: float
    ) -> Dict[str, float]:
        """
        Проверяет весь скелет на анатомическую корректность
        Возвращает: {bone_name: confidence_score}
        """
        
        scores = {}
        
        # Проверяем каждое соединение
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
                # Обновляем счёт обеих костей
                scores[bone1_name] = max(scores.get(bone1_name, 0), conf)
                scores[bone2_name] = max(scores.get(bone2_name, 0), conf)
            else:
                # Невалидное соединение - снижаем счёт
                scores[bone2_name] = min(scores.get(bone2_name, 1.0), 0.3)
        
        # Проверяем углы в цепочках
        for chain_name, bone_sequence in LIMB_CHAINS.items():
            if len(bone_sequence) < 3:
                continue
            
            for i in range(len(bone_sequence) - 2):
                b1, b2, b3 = bone_sequence[i], bone_sequence[i+1], bone_sequence[i+2]
                
                if b1 in bones and b2 in bones and b3 in bones:
                    valid, conf = AnatomicalValidator.validate_angle(
                        b1, bones[b1].position,
                        b2, bones[b2].position,
                        b3, bones[b3].position
                    )
                    
                    if valid:
                        scores[b2] = max(scores.get(b2, 0), conf)
        
        return scores


# =============================================================================
# MULTI-CHANNEL SKELETON BUILDER (многоканальный поиск костей)
# =============================================================================

@dataclass
class BoneCandidate:
    """Кандидат на кость из разных источников"""
    bone_name: str
    position: Tuple[float, float]
    confidence: float
    depth: float
    source: str  # "dwpose", "contour", "primitive", "silhouette", "estimated"
    contour: Optional['Contour'] = None
    primitive: Optional[DetectedPrimitive] = None
    
    def merge_with(self, other: 'BoneCandidate') -> 'BoneCandidate':
        """Объединяет два кандидата одной кости"""
        # Взвешенное среднее по уверенности
        total_conf = self.confidence + other.confidence
        
        new_x = (self.position[0] * self.confidence + other.position[0] * other.confidence) / total_conf
        new_y = (self.position[1] * self.confidence + other.position[1] * other.confidence) / total_conf
        new_depth = (self.depth * self.confidence + other.depth * other.confidence) / total_conf
        
        # Источник - комбинированный
        sources = [self.source, other.source]
        new_source = "+".join(sorted(set(sources)))
        
        # Уверенность - усиленная
        new_conf = min(1.0, total_conf * 0.7)  # Бонус за согласованность
        
        return BoneCandidate(
            bone_name=self.bone_name,
            position=(new_x, new_y),
            confidence=new_conf,
            depth=new_depth,
            source=new_source,
            contour=self.contour or other.contour,
            primitive=self.primitive or other.primitive
        )


class MultiChannelSkeletonBuilder:
    """
    Многоканальный построитель скелета:
    1. DWPose (базовые ключевые точки)
    2. Геометрические примитивы (форма контуров)
    3. Силуэт (выпуклые области)
    4. Анатомическая оценка (пропорции)
    5. Двунаправленный поиск (цепочки)
    """
    
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
        contours: List['Contour'],
        depth_map: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        body_mask: Optional[np.ndarray],
        enabled_bones: Set[str],
        img_size: Tuple[int, int]
    ) -> Tuple[Dict[str, BoneResult], Dict[str, Any]]:
        """
        Строит скелет из всех доступных источников
        Возвращает: (bones, metadata)
        """
        
        h, w = img_size
        
        # === ШАГ 1: АНАЛИЗ ПЕРСПЕКТИВЫ И ПРИОРИТЕТОВ ===
        head_pos = None
        if "Head" in dwpose_kp and dwpose_kp["Head"][2] > 0.3:
            head_pos = (dwpose_kp["Head"][0], dwpose_kp["Head"][1])
        
        if not head_pos:
            debug_print("ERROR: No head position, cannot build skeleton")
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
        
        # === ШАГ 2: АНАЛИЗ СИЛУЭТА И ВЫПУКЛЫХ ОБЛАСТЕЙ ===
        if normals is not None and body_mask is not None:
            self.silhouette = SilhouetteAnalyzer.extract_silhouette(
                normals, body_mask, depth_map
            )
            
            self.convex_regions = SilhouetteAnalyzer.find_convex_regions(
                normals, depth_map, body_mask, img_size
            )
        
        # === ШАГ 3: ДЕТЕКЦИЯ ГЕОМЕТРИЧЕСКИХ ПРИМИТИВОВ ===
        self.primitives = GeometricPrimitiveDetector.detect_primitives(
            contours, self.convex_regions, self.perspective, img_size
        )
        
        # === ШАГ 4: СБОР КАНДИДАТОВ ИЗ ВСЕХ ИСТОЧНИКОВ ===
        all_candidates = defaultdict(list)
        
        # Канал 1: DWPose
        for bone_name, kp in dwpose_kp.items():
            if kp[2] > 0.3 and bone_name in enabled_bones:
                depth = self._get_depth(kp[0], kp[1], depth_map, w, h)
                all_candidates[bone_name].append(BoneCandidate(
                    bone_name=bone_name,
                    position=(kp[0], kp[1]),
                    confidence=kp[2] * 0.9,  # Высокая уверенность для DWPose
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
        
        # Канал 4: Анатомическая оценка (для недостающих костей)
        estimated = self._estimate_missing_bones(
            all_candidates, enabled_bones, depth_map, w, h
        )
        for bone_name, candidate in estimated.items():
            all_candidates[bone_name].append(candidate)
        
        debug_print(f"Collected candidates from all channels:")
        for bone_name, candidates in all_candidates.items():
            sources = [c.source for c in candidates]
            debug_print(f"  {bone_name}: {len(candidates)} candidates from {sources}")
        
        # === ШАГ 5: ОБЪЕДИНЕНИЕ КАНДИДАТОВ ===
        merged_candidates = {}
        for bone_name, candidates in all_candidates.items():
            if len(candidates) == 1:
                merged_candidates[bone_name] = candidates[0]
            else:
                # Объединяем близкие кандидаты
                merged = self._merge_candidates(candidates)
                merged_candidates[bone_name] = merged
        
        # === ШАГ 6: ПОСТРОЕНИЕ ЦЕПОЧЕК С ПРИОРИТЕТАМИ ===
        bones = self._build_chains_with_priorities(
            merged_candidates, enabled_bones, depth_map, img_size
        )
        
        # === ШАГ 7: АНАТОМИЧЕСКАЯ ВАЛИДАЦИЯ ===
        connections = self._get_connections(bones, enabled_bones)
        validation_scores = AnatomicalValidator.validate_skeleton(
            bones, connections, self.head_props.head_size
        )
        
        # Обновляем уверенность на основе валидации
        for bone_name, score in validation_scores.items():
            if bone_name in bones:
                bones[bone_name].confidence *= (0.7 + score * 0.3)  # Модуляция
        
        debug_print(f"Final skeleton: {len(bones)} bones")
        debug_print(f"Validation scores: {validation_scores}")
        
        # === ШАГ 8: ДОБАВЛЕНИЕ ПРОИЗВОДНЫХ КОСТЕЙ ===
        self._add_derived_bones(bones, enabled_bones, depth_map, w, h)
        
        # Метаданные для визуализации
        metadata = {
            "perspective": self.perspective,
            "primitives": self.primitives,
            "convex_regions": self.convex_regions,
            "silhouette": self.silhouette,
            "validation_scores": validation_scores,
            "all_candidates": all_candidates,
        }
        
        return bones, metadata
    
    def _match_primitives_to_bones(
        self,
        primitives: List[DetectedPrimitive],
        enabled_bones: Set[str],
        depth_map: Optional[np.ndarray],
        w: int, h: int
    ) -> Dict[str, BoneCandidate]:
        """Сопоставляет геометрические примитивы с костями"""
        
        candidates = {}
        
        for prim in primitives:
            # Голова - ELLIPSE или CIRCLE, крупная
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
            
            # Торс - TRAPEZOID или RECTANGLE
            elif prim.primitive_type in [GeometricPrimitive.TRAPEZOID, GeometricPrimitive.RECTANGLE]:
                if "Neck" in enabled_bones and "Neck" not in candidates:
                    # Шея - верхняя часть торса
                    neck_y = prim.center[1] - prim.size[1] * 0.4
                    candidates["Neck"] = BoneCandidate(
                        bone_name="Neck",
                        position=(prim.center[0], neck_y),
                        confidence=prim.confidence * 0.7,
                        depth=self._get_depth(prim.center[0], neck_y, depth_map, w, h),
                        source="primitive",
                        primitive=prim
                    )
            
            # Конечности - CONE
            elif prim.primitive_type == GeometricPrimitive.CONE:
                # Определяем тип по размеру и позиции
                size = max(prim.size)
                y_pos = prim.center[1]
                
                # Бедро/голень - нижняя часть, крупнее
                if y_pos > 0.5 and size > 0.08:
                    # Пытаемся определить сторону
                    side = "L" if prim.center[0] < 0.5 else "R"
                    
                    if y_pos < 0.7:
                        bone_name = f"Thigh_{side}"
                    else:
                        bone_name = f"Calf_{side}"
                    
                    if bone_name in enabled_bones and bone_name not in candidates:
                        candidates[bone_name] = BoneCandidate(
                            bone_name=bone_name,
                            position=prim.center,
                            confidence=prim.confidence * 0.75,
                            depth=self._get_depth(prim.center[0], prim.center[1], depth_map, w, h),
                            source="primitive",
                            primitive=prim
                        )
                
                # Плечо/предплечье - верхняя часть
                elif y_pos < 0.6 and size > 0.06:
                    side = "L" if prim.center[0] < 0.5 else "R"
                    
                    if y_pos < 0.4:
                        bone_name = f"Shoulder_{side}"
                    else:
                        bone_name = f"Forearm_{side}"
                    
                    if bone_name in enabled_bones and bone_name not in candidates:
                        candidates[bone_name] = BoneCandidate(
                            bone_name=bone_name,
                            position=prim.center,
                            confidence=prim.confidence * 0.75,
                            depth=self._get_depth(prim.center[0], prim.center[1], depth_map, w, h),
                            source="primitive",
                            primitive=prim
                        )
            
            # Круглые суставы или пятка - CIRCLE
            elif prim.primitive_type == GeometricPrimitive.CIRCLE:
                # Пятка - внизу
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
    
    def _match_convex_to_bones(
        self,
        convex_regions: List[Tuple],
        enabled_bones: Set[str],
        depth_map: Optional[np.ndarray],
        w: int, h: int
    ) -> Dict[str, BoneCandidate]:
        """Сопоставляет выпуклые области с костями"""
        
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
            
            elif region_type == "limb":
                # Определяем тип конечности по позиции
                y_pos = center[1]
                side = "L" if center[0] < 0.5 else "R"
                
                if y_pos > 0.6:
                    bone_name = f"Calf_{side}"
                elif y_pos > 0.4:
                    bone_name = f"Thigh_{side}"
                elif y_pos > 0.3:
                    bone_name = f"Forearm_{side}"
                else:
                    bone_name = f"Shoulder_{side}"
                
                if bone_name in enabled_bones and bone_name not in candidates:
                    candidates[bone_name] = BoneCandidate(
                        bone_name=bone_name,
                        position=center,
                        confidence=0.6,
                        depth=self._get_depth(center[0], center[1], depth_map, w, h),
                        source="silhouette"
                    )
        
        return candidates
    
    def _estimate_missing_bones(
        self,
        existing_candidates: Dict[str, List[BoneCandidate]],
        enabled_bones: Set[str],
        depth_map: Optional[np.ndarray],
        w: int, h: int
    ) -> Dict[str, BoneCandidate]:
        """Оценивает позиции недостающих костей по анатомическим пропорциям"""
        
        estimated = {}
        
        # Если есть голова, оцениваем шею
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
        
        # Если есть шея, оцениваем плечи
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
        
        # Если есть плечо, оцениваем предплечье
        for side in ["L", "R"]:
            shoulder_key = f"Shoulder_{side}"
            forearm_key = f"Forearm_{side}"
            
            if shoulder_key in existing_candidates and forearm_key not in existing_candidates and forearm_key in enabled_bones:
                shoulder_pos = existing_candidates[shoulder_key][0].position
                
                # Направление вниз и в сторону
                offset_x = -0.1 if side == "L" else 0.1
                forearm_pos = (shoulder_pos[0] + offset_x, shoulder_pos[1] + 0.25)
                
                estimated[forearm_key] = BoneCandidate(
                    bone_name=forearm_key,
                    position=forearm_pos,
                    confidence=0.4,
                    depth=self._get_depth(forearm_pos[0], forearm_pos[1], depth_map, w, h),
                    source="estimated"
                )
        
        # Аналогично для ног (если есть таз или шея)
        pelvis_pos = None
        if "Pelvis" in existing_candidates:
            pelvis_pos = existing_candidates["Pelvis"][0].position
        elif "Neck" in existing_candidates:
            neck_pos = existing_candidates["Neck"][0].position
            pelvis_pos = (neck_pos[0], neck_pos[1] + self.head_props.head_size * 3.5)
        
        if pelvis_pos:
            for side in ["L", "R"]:
                thigh_key = f"Thigh_{side}"
                if thigh_key not in existing_candidates and thigh_key in enabled_bones:
                    offset_x = -0.08 if side == "L" else 0.08
                    thigh_pos = (pelvis_pos[0] + offset_x, pelvis_pos[1] + 0.15)
                    
                    estimated[thigh_key] = BoneCandidate(
                        bone_name=thigh_key,
                        position=thigh_pos,
                        confidence=0.4,
                        depth=self._get_depth(thigh_pos[0], thigh_pos[1], depth_map, w, h),
                        source="estimated"
                    )
        
        return estimated
    
    def _merge_candidates(self, candidates: List[BoneCandidate]) -> BoneCandidate:
        """Объединяет несколько кандидатов одной кости"""
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Группируем близкие кандидаты
        # Если позиции близки (< 0.05), объединяем
        groups = []
        
        for candidate in candidates:
            added = False
            for group in groups:
                # Проверяем близость к любому в группе
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
        
        # Выбираем группу с максимальной суммарной уверенностью
        best_group = max(groups, key=lambda g: sum(c.confidence for c in g))
        
        # Объединяем группу
        result = best_group[0]
        for candidate in best_group[1:]:
            result = result.merge_with(candidate)
        
        return result
    
    def _build_chains_with_priorities(
        self,
        candidates: Dict[str, BoneCandidate],
        enabled_bones: Set[str],
        depth_map: Optional[np.ndarray],
        img_size: Tuple[int, int]
    ) -> Dict[str, BoneResult]:
        """Строит цепочки с учётом приоритетов перспективы"""
        
        bones = {}
        
        # Сортируем кандидатов по приоритету
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
    
    def _add_derived_bones(
        self,
        bones: Dict[str, BoneResult],
        enabled_bones: Set[str],
        depth_map: Optional[np.ndarray],
        w: int, h: int
    ):
        """Добавляет производные кости (Spine, Clavicles, etc.)"""
        
        # Spine от шеи к тазу
        neck = bones.get("Neck")
        pelvis = bones.get("Pelvis")
        
        if not pelvis and ("Thigh_L" in bones or "Thigh_R" in bones):
            # Оцениваем таз от бёдер
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
        
        # Root от таза
        if pelvis and "Root" not in bones and "Root" in enabled_bones:
            bones["Root"] = BoneResult(
                bone_name="Root",
                position=pelvis.position,
                confidence=pelvis.confidence * 0.9,
                depth=pelvis.depth,
                contour=None
            )
        
        # Clavicles от шеи к плечам
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
    
    def _get_connections(self, bones: Dict[str, BoneResult], enabled_bones: Set[str]) -> List[Tuple[str, str]]:
        """Возвращает соединения костей"""
        connections = []
        
        for bone_name in bones.keys():
            bone_def = BONE_DEFINITIONS.get(bone_name)
            if bone_def and bone_def.parent:
                if bone_def.parent in bones:
                    connections.append((bone_def.parent, bone_name))
        
        # Fallback для ног если нет Pelvis
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
# BODY TILT ANALYZER (анализ наклона тела по глубине)
# =============================================================================

@dataclass
class BodyTilt:
    """Наклон тела определённый по градиенту глубины"""
    angle: float  # угол наклона в градусах (-90 до 90, 0 = вертикально)
    direction: str  # "forward", "backward", "left", "right", "upright"
    confidence: float
    head_depth: float
    torso_depth: float
    legs_depth: float
    
    def get_search_modification(self) -> Tuple[float, float]:
        """
        Возвращает модификацию направления поиска (dx, dy)
        в зависимости от наклона
        """
        if self.direction == "forward":
            # Наклон вперёд - голова ниже таза по глубине (темнее)
            return (0, 0.2)  # Смещаем поиск ног немного вперёд
        elif self.direction == "backward":
            # Наклон назад - голова выше (светлее)
            return (0, -0.1)
        return (0, 0)


class BodyTiltAnalyzer:
    """Анализ наклона тела по карте глубины"""
    
    @staticmethod
    def analyze(
        head_pos: Tuple[float, float],
        neck_pos: Optional[Tuple[float, float]],
        pelvis_pos: Optional[Tuple[float, float]],
        depth_map: np.ndarray,
        img_size: Tuple[int, int]
    ) -> BodyTilt:
        """
        Анализирует наклон тела по градиенту глубины
        Светлое = ближе к камере, темное = дальше
        """
        if depth_map is None:
            return BodyTilt(0, "upright", 0.3, 0.5, 0.5, 0.5)
        
        h, w = img_size
        dh, dw = depth_map.shape[:2]
        
        # Получаем глубину головы
        hx = int(head_pos[0] * dw)
        hy = int(head_pos[1] * dh)
        head_depth = float(depth_map[
            max(0, min(dh-1, hy)),
            max(0, min(dw-1, hx))
        ])
        
        # Глубина шеи/торса
        if neck_pos:
            nx = int(neck_pos[0] * dw)
            ny = int(neck_pos[1] * dh)
            torso_depth = float(depth_map[
                max(0, min(dh-1, ny)),
                max(0, min(dw-1, nx))
            ])
        else:
            # Примерно ниже головы
            torso_y = int((head_pos[1] + 0.3) * dh)
            torso_depth = float(depth_map[
                max(0, min(dh-1, torso_y)),
                max(0, min(dw-1, hx))
            ])
        
        # Глубина области ног
        if pelvis_pos:
            px = int(pelvis_pos[0] * dw)
            py = int(pelvis_pos[1] * dh)
            legs_depth = float(depth_map[
                max(0, min(dh-1, py)),
                max(0, min(dw-1, px))
            ])
        else:
            # Примерно внизу от головы
            legs_y = int((head_pos[1] + 0.6) * dh)
            legs_depth = float(depth_map[
                max(0, min(dh-1, legs_y)),
                max(0, min(dw-1, hx))
            ])
        
        # Анализ градиента
        head_to_torso = torso_depth - head_depth
        torso_to_legs = legs_depth - torso_depth
        head_to_legs = legs_depth - head_depth
        
        # Определяем наклон
        # Если голова темнее (дальше) ног = наклон вперёд
        # Если голова светлее (ближе) ног = наклон назад
        
        threshold = 0.1
        
        if head_to_legs > threshold:
            # Ноги ближе к камере - наклон вперёд
            direction = "forward"
            angle = min(45, head_to_legs * 90)
            confidence = min(1.0, abs(head_to_legs) * 2)
        elif head_to_legs < -threshold:
            # Голова ближе - наклон назад
            direction = "backward"
            angle = max(-45, head_to_legs * 90)
            confidence = min(1.0, abs(head_to_legs) * 2)
        else:
            # Примерно на одной глубине - вертикально
            direction = "upright"
            angle = 0
            confidence = 0.8
        
        debug_print(f"Body tilt: {direction} (angle: {angle:.1f}°, conf: {confidence:.2f})")
        debug_print(f"  Head depth: {head_depth:.3f}, Torso: {torso_depth:.3f}, Legs: {legs_depth:.3f}")
        
        return BodyTilt(
            angle=angle,
            direction=direction,
            confidence=confidence,
            head_depth=head_depth,
            torso_depth=torso_depth,
            legs_depth=legs_depth
        )


# =============================================================================
# CONVEX NORMALS (нормали поверхности)
# =============================================================================

class ConvexNormalsHelper:
    @staticmethod
    def compute_surface_normals(depth_map: np.ndarray) -> np.ndarray:
        """Вычисляет нормали поверхности из карты глубины"""
        if depth_map is None or depth_map.size == 0:
            return None
        
        # Градиенты глубины (Sobel для лучшего качества)
        gx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        
        # Нормали
        normals = np.zeros((*depth_map.shape, 3), dtype=np.float32)
        normals[:, :, 0] = -gx
        normals[:, :, 1] = -gy
        normals[:, :, 2] = 1.0
        
        # Нормализация
        norm = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True)) + 1e-8
        normals = normals / norm
        
        return normals
    
    @staticmethod
    def compute_convexity(depth_map: np.ndarray) -> np.ndarray:
        """Карта выпуклости (Лапласиан)"""
        if depth_map is None:
            return None
        
        laplacian = cv2.Laplacian(depth_map.astype(np.float32), cv2.CV_32F, ksize=5)
        
        # Нормализация в 0-1
        if laplacian.max() > laplacian.min():
            laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
        
        return laplacian
    
    @staticmethod
    def visualize_normals(normals: np.ndarray) -> np.ndarray:
        """Визуализация нормалей как RGB (для отладки)"""
        if normals is None:
            return None
        
        # Нормали в диапазоне [-1, 1] -> RGB [0, 255]
        vis = ((normals + 1.0) * 127.5).astype(np.uint8)
        return vis


# =============================================================================
# TRAPEZOID TORSO DETECTOR (трапециевидный торс)
# =============================================================================

@dataclass
class TrapezoidTorso:
    """Трапециевидный торс (front view) или прямоугольный (side view)"""
    neck_point: Tuple[float, float]
    shoulder_left: Tuple[float, float]
    shoulder_right: Tuple[float, float]
    waist_left: Tuple[float, float]
    waist_right: Tuple[float, float]
    hip_left: Tuple[float, float]
    hip_right: Tuple[float, float]
    center: Tuple[float, float]
    depth: float
    shape_type: str  # "trapezoid" (front/back) or "rectangle" (side/top/bottom)
    orientation: str  # "front", "back", "side_left", "side_right"
    confidence: float
    contour: Optional['Contour'] = None


class TrapezoidTorsoDetector:
    """
    Детектор торса с учётом формы:
    - Front/Back: трапеция с 2 изгибами над плечами + 1 изгиб под плечами
    - Side/Top/Bottom: прямоугольник
    """
    
    @staticmethod
    def detect_from_contours(
        contours: List['Contour'],
        neck_pos: Tuple[float, float],
        head_size: float,
        depth_map: Optional[np.ndarray],
        face_direction: str,
        img_size: Tuple[int, int]
    ) -> Optional[TrapezoidTorso]:
        """Поиск торса по контурам с учётом формы"""
        
        h, w = img_size
        
        # Ожидаемые параметры торса
        expected_width = head_size * 2.5  # Плечи шире головы
        expected_height = head_size * 3.0  # От шеи до таза
        
        # Область поиска - ниже шеи
        search_y_min = neck_pos[1]
        search_y_max = min(1.0, neck_pos[1] + expected_height)
        search_x_min = max(0.0, neck_pos[0] - expected_width / 2)
        search_x_max = min(1.0, neck_pos[0] + expected_width / 2)
        
        candidates = []
        
        for cnt in contours:
            # Проверка попадания в область
            if not (search_y_min <= cnt.center[1] <= search_y_max and
                    search_x_min <= cnt.center[0] <= search_x_max):
                continue
            
            # Проверка размера
            cnt_size = max(cnt.size)
            if not (head_size * 1.5 <= cnt_size <= head_size * 4.0):
                continue
            
            # Анализ формы
            shape_score = TrapezoidTorsoDetector._analyze_torso_shape(
                cnt, neck_pos, face_direction, img_size
            )
            
            if shape_score > 0.3:
                candidates.append((cnt, shape_score))
        
        if not candidates:
            debug_print("No torso contour found, using estimation")
            return TrapezoidTorsoDetector._estimate_torso(
                neck_pos, head_size, depth_map, face_direction, img_size
            )
        
        # Лучший кандидат
        best_cnt, best_score = max(candidates, key=lambda x: x[1])
        
        debug_print(f"Found torso contour with score {best_score:.2f}")
        
        return TrapezoidTorsoDetector._build_torso_from_contour(
            best_cnt, neck_pos, depth_map, face_direction, img_size
        )
    
    @staticmethod
    def _analyze_torso_shape(
        contour: 'Contour',
        neck_pos: Tuple[float, float],
        face_direction: str,
        img_size: Tuple[int, int]
    ) -> float:
        """Анализ формы контура на соответствие торсу"""
        
        score = 0.0
        
        # 1. Позиция - должен быть ниже шеи
        if contour.center[1] > neck_pos[1]:
            score += 0.2
        
        # 2. Соотношение сторон
        aspect = min(contour.size) / max(contour.size) if max(contour.size) > 0 else 0
        
        if face_direction in ["front", "back"]:
            # Front/back: трапеция - более вытянутая вертикально
            if 0.3 <= aspect <= 0.7:
                score += 0.3
        else:
            # Side: прямоугольник - может быть более квадратным
            if 0.5 <= aspect <= 0.9:
                score += 0.3
        
        # 3. Анализ вогнутостей (для трапеции)
        if contour.points is not None and len(contour.points) >= 10:
            hull = cv2.convexHull(contour.points, returnPoints=False)
            try:
                defects = cv2.convexityDefects(contour.points, hull)
                if defects is not None:
                    # Для трапеции ожидаем 2-3 значительных вогнутости
                    # (изгибы над плечами и под плечами)
                    significant = sum(1 for d in defects if d[0][3] > 2000)
                    
                    if face_direction in ["front", "back"]:
                        if 2 <= significant <= 4:
                            score += 0.3
                    else:
                        # Side view - меньше вогнутостей
                        if significant <= 2:
                            score += 0.2
            except Exception:
                pass
        
        # 4. Расстояние от шеи
        dist = math.sqrt(
            (contour.center[0] - neck_pos[0])**2 +
            (contour.center[1] - neck_pos[1])**2
        )
        dist_score = max(0, 1.0 - dist * 2)
        score += dist_score * 0.2
        
        return score
    
    @staticmethod
    def _build_torso_from_contour(
        contour: 'Contour',
        neck_pos: Tuple[float, float],
        depth_map: Optional[np.ndarray],
        face_direction: str,
        img_size: Tuple[int, int]
    ) -> TrapezoidTorso:
        """Построение торса из контура"""
        
        h, w = img_size
        
        # Получаем bounding box контура
        if contour.cv_ellipse:
            (cx, cy), (rw, rh), angle = contour.cv_ellipse
            cx_n, cy_n = cx / w, cy / h
            rw_n, rh_n = rw / w, rh / h
        else:
            cx_n, cy_n = contour.center
            rw_n, rh_n = contour.size
        
        # Определяем ключевые точки трапеции
        half_w_top = rw_n * 0.5  # Плечи
        half_w_mid = rw_n * 0.4  # Талия
        half_w_bot = rw_n * 0.45  # Бёдра
        
        # Плечи (сверху)
        shoulder_y = neck_pos[1] + rh_n * 0.15
        shoulder_l = (cx_n - half_w_top, shoulder_y)
        shoulder_r = (cx_n + half_w_top, shoulder_y)
        
        # Талия (середина)
        waist_y = neck_pos[1] + rh_n * 0.5
        waist_l = (cx_n - half_w_mid, waist_y)
        waist_r = (cx_n + half_w_mid, waist_y)
        
        # Бёдра (низ)
        hip_y = neck_pos[1] + rh_n * 0.85
        hip_l = (cx_n - half_w_bot, hip_y)
        hip_r = (cx_n + half_w_bot, hip_y)
        
        center = (cx_n, cy_n)
        
        # Глубина
        if depth_map is not None:
            dh, dw = depth_map.shape[:2]
            dx = int(cx_n * dw)
            dy = int(cy_n * dh)
            depth = float(depth_map[
                max(0, min(dh-1, dy)),
                max(0, min(dw-1, dx))
            ])
        else:
            depth = 0.5
        
        # Форма
        shape_type = "trapezoid" if face_direction in ["front", "back"] else "rectangle"
        
        return TrapezoidTorso(
            neck_point=neck_pos,
            shoulder_left=shoulder_l,
            shoulder_right=shoulder_r,
            waist_left=waist_l,
            waist_right=waist_r,
            hip_left=hip_l,
            hip_right=hip_r,
            center=center,
            depth=depth,
            shape_type=shape_type,
            orientation=face_direction,
            confidence=0.7,
            contour=contour
        )
    
    @staticmethod
    def _estimate_torso(
        neck_pos: Tuple[float, float],
        head_size: float,
        depth_map: Optional[np.ndarray],
        face_direction: str,
        img_size: Tuple[int, int]
    ) -> TrapezoidTorso:
        """Оценка торса если контур не найден"""
        
        # Стандартные пропорции
        shoulder_width = head_size * 1.5
        waist_width = head_size * 1.2
        hip_width = head_size * 1.3
        torso_height = head_size * 3.0
        
        shoulder_y = neck_pos[1] + head_size * 0.3
        waist_y = neck_pos[1] + head_size * 1.5
        hip_y = neck_pos[1] + head_size * 2.5
        
        center = (neck_pos[0], neck_pos[1] + torso_height / 2)
        
        if depth_map is not None:
            h, w = img_size
            dh, dw = depth_map.shape[:2]
            dx = int(center[0] * dw)
            dy = int(center[1] * dh)
            depth = float(depth_map[
                max(0, min(dh-1, dy)),
                max(0, min(dw-1, dx))
            ])
        else:
            depth = 0.5
        
        shape_type = "trapezoid" if face_direction in ["front", "back"] else "rectangle"
        
        return TrapezoidTorso(
            neck_point=neck_pos,
            shoulder_left=(neck_pos[0] - shoulder_width/2, shoulder_y),
            shoulder_right=(neck_pos[0] + shoulder_width/2, shoulder_y),
            waist_left=(neck_pos[0] - waist_width/2, waist_y),
            waist_right=(neck_pos[0] + waist_width/2, waist_y),
            hip_left=(neck_pos[0] - hip_width/2, hip_y),
            hip_right=(neck_pos[0] + hip_width/2, hip_y),
            center=center,
            depth=depth,
            shape_type=shape_type,
            orientation=face_direction,
            confidence=0.4,
            contour=None
        )

# =============================================================================
# FACE DIRECTION ANALYZER
# =============================================================================

class FaceDirection(Enum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


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
        """Направление поиска шеи от лица"""
        return (0.0, 1.0)
    
    def get_shoulder_direction(self, side: str) -> Tuple[float, float]:
        """Направление поиска плеча от шеи"""
        if side == "L":
            if self.direction == FaceDirection.BACK:
                return (-0.7, 0.3)
            else:
                return (-0.6, 0.4)
        else:
            if self.direction == FaceDirection.BACK:
                return (0.7, 0.3)
            else:
                return (0.6, 0.4)


class FaceDirectionAnalyzer:
    @staticmethod
    def analyze(face_keypoints: List[Tuple[float, float, float]], head_pos: Tuple[float, float]) -> FaceAnalysis:
        """Анализирует направление лица"""
        
        if not face_keypoints or len(face_keypoints) < 5:
            return FaceAnalysis(
                center=head_pos,
                direction=FaceDirection.FRONT,
                confidence=0.3,
                size=0.12
            )
        
        # Собираем точки
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
        
        # Простое определение направления по симметрии
        direction = FaceDirection.FRONT
        confidence = 0.7
        
        return FaceAnalysis(
            center=face_center,
            direction=direction,
            confidence=confidence,
            size=face_size
        )


# =============================================================================
# CONVEX NORMALS HELPER
# =============================================================================

class ConvexNormalsHelper:
    @staticmethod
    def compute_surface_normals(depth_map: np.ndarray) -> np.ndarray:
        if depth_map is None or depth_map.size == 0:
            return None
        
        gy, gx = np.gradient(depth_map.astype(np.float32))
        normals = np.zeros((*depth_map.shape, 3), dtype=np.float32)
        normals[:, :, 0] = -gx
        normals[:, :, 1] = -gy
        normals[:, :, 2] = 1.0
        norm = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True)) + 1e-8
        normals = normals / norm
        return normals
    
    @staticmethod
    def compute_convexity(depth_map: np.ndarray) -> np.ndarray:
        if depth_map is None:
            return None
        
        laplacian = cv2.Laplacian(depth_map.astype(np.float32), cv2.CV_32F)
        if laplacian.max() > laplacian.min():
            laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
        return laplacian
    
    @staticmethod
    def is_convex_region(
        center: Tuple[float, float],
        depth_map: np.ndarray,
        convexity_map: np.ndarray,
        img_size: Tuple[int, int],
        threshold: float = 0.4
    ) -> bool:
        if depth_map is None or convexity_map is None:
            return True
        
        h, w = img_size
        cx = int(center[0] * w)
        cy = int(center[1] * h)
        
        radius = 5
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius + 1)
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius + 1)
        
        region = convexity_map[y1:y2, x1:x2]
        if region.size == 0:
            return True
        
        avg_convexity = region.mean()
        return avg_convexity > threshold
    
    @staticmethod
    def get_body_depth_range(
        depth_map: np.ndarray,
        body_mask: np.ndarray,
        percentile_margin: float = 10
    ) -> Tuple[float, float]:
        if depth_map is None or body_mask is None:
            return 0.0, 1.0
        
        body_pixels = depth_map[body_mask > 128]
        if len(body_pixels) == 0:
            return 0.0, 1.0
        
        d_min = np.percentile(body_pixels, percentile_margin)
        d_max = np.percentile(body_pixels, 100 - percentile_margin)
        return float(d_min), float(d_max)


# =============================================================================
# HEAD PROPORTIONS
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
    
    def get_search_direction_vector(self, bone_name: str, parent_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Вектор направления поиска от родителя"""
        if bone_name not in HEAD_BODY_PROPORTIONS:
            return (0.0, 1.0)
        
        prop = HEAD_BODY_PROPORTIONS[bone_name]
        dx = prop.offset_from_head[0] * self.head_width
        dy = prop.offset_from_head[1] * self.head_height
        
        # Нормализация
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            return (dx/length, dy/length)
        return (0.0, 1.0)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class LimbPartType(Enum):
    HEAD = "head"
    NECK = "neck"
    TORSO = "torso"
    THIGH = "thigh"
    CALF = "calf"
    FOOT = "foot"
    CLAVICLE = "clavicle"
    UPPER_ARM = "upper_arm"
    FOREARM = "forearm"
    HAND = "hand"
    UNKNOWN = "unknown"


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
    """Результат поиска кости"""
    bone_name: str
    position: Tuple[float, float]
    confidence: float
    depth: float
    contour: Optional[Contour]


@dataclass
class LimbChain:
    """Цепочка конечности"""
    chain_name: str  # "arm_L", "arm_R", "leg_L", "leg_R"
    bones: Dict[str, BoneResult]  # bone_name -> BoneResult
    start_pos: Tuple[float, float]
    confidence: float


@dataclass
class ConeTorso:
    neck_point: Tuple[float, float]
    shoulder_left: Tuple[float, float]
    shoulder_right: Tuple[float, float]
    hip_left: Tuple[float, float]
    hip_right: Tuple[float, float]
    center: Tuple[float, float]
    depth: float
    orientation: str
    confidence: float


# =============================================================================
# PROCESSORS
# =============================================================================

class DepthProcessor:
    def __init__(self, model_name: str = "midas_v21_small"):
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self.device = None
    
    def _get_best_device(self):
        """Определяет лучшее устройство для PyTorch"""
        if self.device is not None:
            return self.device
        
        try:
            # DirectML для AMD на Windows
            import torch_directml
            self.device = torch_directml.device()
            debug_print(f"Using DirectML device: {self.device}")
            return self.device
        except ImportError:
            pass
        
        # CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            debug_print(f"Using CUDA device")
            return self.device
        
        # ROCm (использует CUDA API)
        if hasattr(torch, 'hip') and torch.hip.is_available():
            self.device = torch.device("cuda")
            debug_print(f"Using ROCm device")
            return self.device
        
        # CPU fallback
        self.device = torch.device("cpu")
        debug_print(f"Using CPU device")
        return self.device
    
    def load_model(self):
        if self._loaded:
            return
        try:
            # ИСПРАВЛЕНО: используем локальный путь к custom_controlnet_aux
            import sys
            import os
            
            # Путь к custom_controlnet_aux в ComfyUI
            controlnet_aux_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "..", 
                "custom_controlnet_aux"
            )
            
            if os.path.exists(controlnet_aux_path):
                sys.path.insert(0, controlnet_aux_path)
            
            # Импорт MiDaS
            try:
                from midas import MidasDetector
                self.model = MidasDetector.from_pretrained("lllyasviel/Annotators")
                debug_print(f"MiDaS loaded successfully")
            except ImportError:
                # Альтернативный путь
                from custom_midas import MidasDetector
                self.model = MidasDetector.from_pretrained()
                debug_print(f"MiDaS loaded (alternative)")
            
            # Перемещаем на устройство
            device = self._get_best_device()
            if hasattr(self.model, 'model') and device is not None:
                try:
                    self.model.model.to(device)
                    debug_print(f"Depth model moved to {device}")
                except Exception as e:
                    debug_print(f"Failed to move model to device: {e}")
                    
        except Exception as e:
            debug_print(f"Depth model load failed: {e}")
            debug_print(f"Depth processing will be disabled")
            self.model = None
        
        self._loaded = True
    
    def process(self, image: np.ndarray) -> Optional[np.ndarray]:
        if not self._loaded:
            self.load_model()
        
        if self.model is None:
            debug_print("Depth model not available, skipping depth processing")
            return None
        
        try:
            depth = self.model(image)
            if isinstance(depth, tuple):
                depth = depth[0]
            
            # Конвертируем в grayscale если нужно
            if depth.ndim == 3:
                depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
            
            depth = depth.astype(np.float32)
            
            # Нормализация
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
                
                # Путь к custom_controlnet_aux
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
    def __init__(self, model_name: str = "canny"):
        self.model_name = model_name
        self.model = None
        self._loaded = False
    
    def load_model(self):
        if self._loaded:
            return
        if self.model_name != "canny":
            try:
                if self.model_name == "hed":
                    from controlnet_aux import HEDdetector
                    self.model = HEDdetector.from_pretrained("lllyasviel/Annotators")
            except Exception:
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
        except Exception:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            return cv2.Canny(gray, low, high)
    
    def unload(self):
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
            body_depth_range = ConvexNormalsHelper.get_body_depth_range(depth_map, body_mask)
        
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
                
                # Фильтры
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
                
                if convexity_map is not None:
                    if not ConvexNormalsHelper.is_convex_region(
                        (cx_n, cy_n), depth_map, convexity_map, (h, w), threshold=0.3
                    ):
                        continue
                
                cv_ellipse = None
                try:
                    cv_ellipse = cv2.fitEllipse(cnt)
                except Exception:
                    pass
                
                depth_grad = 0.0
                
                contours.append(Contour(
                    idx=idx,
                    points=cnt,
                    center=(cx_n, cy_n),
                    size=(rw_n, rh_n),
                    angle=angle,
                    area=area / (w * h),
                    depth=depth_val,
                    depth_gradient=depth_grad,
                    cv_ellipse=cv_ellipse
                ))
            except Exception:
                continue
        
        debug_print(f"Found {len(contours)} valid contours")
        return contours

# =============================================================================
# BIDIRECTIONAL CHAIN SEARCHER (двунаправленный поиск)
# =============================================================================

class BidirectionalChainSearcher:
    """
    Двунаправленный поиск конечностей:
    1. От торса → к конечностям
    2. От конечностей → к торсу
    3. Выбор лучшего варианта соединения
    """
    
    def __init__(self):
        self.limb_chains = []
        self.head_props = None
        self.face_analysis = None
        self.body_tilt = None
    
    def search_chains(
        self,
        contours: List[Contour],
        face_analysis: FaceAnalysis,
        head_pos: Tuple[float, float],
        dwpose_kp: Dict[str, Tuple],
        depth_map: Optional[np.ndarray],
        convexity_map: Optional[np.ndarray],
        enabled_bones: Set[str],
        img_size: Tuple[int, int]
    ) -> Tuple[Dict[str, BoneResult], List[LimbChain], TrapezoidTorso, BodyTilt]:
        """Двунаправленный поиск всех цепочек"""
        
        self.face_analysis = face_analysis
        h, w = img_size
        
        # Пропорции головы
        self.head_props = HeadProportions(
            head_center=head_pos,
            head_size=face_analysis.size,
            head_width=face_analysis.size * 0.8,
            head_height=face_analysis.size * 1.2
        )
        
        all_bones = {}
        
        # Голова
        all_bones["Head"] = BoneResult(
            bone_name="Head",
            position=head_pos,
            confidence=0.9,
            depth=self._get_depth(head_pos[0], head_pos[1], depth_map, w, h),
            contour=None
        )
        
        # Шея (от головы вниз)
        neck_result = self._search_bone(
            "Neck",
            head_pos,
            face_analysis.get_neck_search_direction(),
            contours,
            dwpose_kp,
            depth_map,
            img_size,
            set()
        )
        
        if neck_result:
            all_bones["Neck"] = neck_result
            neck_pos = neck_result.position
        else:
            # Fallback
            neck_pos = (head_pos[0], head_pos[1] + self.head_props.head_size * 0.9)
            all_bones["Neck"] = BoneResult(
                bone_name="Neck",
                position=neck_pos,
                confidence=0.5,
                depth=self._get_depth(neck_pos[0], neck_pos[1], depth_map, w, h),
                contour=None
            )
        
        # Анализ наклона тела по глубине
        self.body_tilt = BodyTiltAnalyzer.analyze(
            head_pos,
            neck_pos,
            None,  # Таз пока неизвестен
            depth_map,
            img_size
        )
        
        # Поиск торса (трапеция или прямоугольник)
        torso = TrapezoidTorsoDetector.detect_from_contours(
            contours,
            neck_pos,
            self.head_props.head_size,
            depth_map,
            face_analysis.direction.value,
            img_size
        )
        
        debug_print(f"Torso found: {torso.shape_type}, orientation: {torso.orientation}")
        
        # === ДВУНАПРАВЛЕННЫЙ ПОИСК РУК ===
        arm_chains = []
        for side in ["L", "R"]:
            chain_name = f"arm_{side}"
            
            # 1. Поиск ОТ плеча (торса)
            shoulder_pos = torso.shoulder_left if side == "L" else torso.shoulder_right
            direction = face_analysis.get_shoulder_direction(side)
            
            chain_from_torso = self._search_chain(
                chain_name,
                LIMB_CHAINS[chain_name],
                shoulder_pos,
                direction,
                contours,
                dwpose_kp,
                depth_map,
                enabled_bones,
                img_size,
                set()
            )
            
            # 2. Поиск ОТ кисти (если есть в DWPose)
            hand_key = f"Hand_{side}"
            chain_from_hand = None
            
            if hand_key in dwpose_kp and dwpose_kp[hand_key][2] > 0.5:
                hand_pos = (dwpose_kp[hand_key][0], dwpose_kp[hand_key][1])
                reversed_chain = list(reversed(LIMB_CHAINS[chain_name]))
                
                chain_from_hand = self._search_chain_reverse(
                    chain_name,
                    reversed_chain,
                    hand_pos,
                    (-direction[0], -direction[1]),  # обратное направление
                    contours,
                    dwpose_kp,
                    depth_map,
                    enabled_bones,
                    img_size,
                    set()
                )
            
            # 3. Выбираем лучший вариант
            best_chain = self._select_best_chain(
                chain_from_torso,
                chain_from_hand,
                shoulder_pos,
                torso
            )
            
            if best_chain:
                arm_chains.append(best_chain)
                all_bones.update(best_chain.bones)
        
        # === ДВУНАПРАВЛЕННЫЙ ПОИСК НОГ ===
        # Сначала обновляем наклон тела с учетом торса
        self.body_tilt = BodyTiltAnalyzer.analyze(
            head_pos,
            neck_pos,
            torso.center,
            depth_map,
            img_size
        )
        
        leg_chains = []
        for side in ["L", "R"]:
            chain_name = f"leg_{side}"
            
            # 1. Поиск ОТ бедра (торса)
            hip_pos = torso.hip_left if side == "L" else torso.hip_right
            
            # Направление с учетом наклона тела
            base_direction = (-0.3 if side == "L" else 0.3, 1.0)
            tilt_mod = self.body_tilt.get_search_modification()
            direction = (base_direction[0] + tilt_mod[0], base_direction[1] + tilt_mod[1])
            
            chain_from_hip = self._search_chain(
                chain_name,
                LIMB_CHAINS[chain_name],
                hip_pos,
                direction,
                contours,
                dwpose_kp,
                depth_map,
                enabled_bones,
                img_size,
                set()
            )
            
            # 2. Поиск ОТ стопы (если есть в DWPose)
            foot_key = f"Foot_{side}"
            chain_from_foot = None
            
            if foot_key in dwpose_kp and dwpose_kp[foot_key][2] > 0.5:
                foot_pos = (dwpose_kp[foot_key][0], dwpose_kp[foot_key][1])
                reversed_chain = list(reversed(LIMB_CHAINS[chain_name]))
                
                chain_from_foot = self._search_chain_reverse(
                    chain_name,
                    reversed_chain,
                    foot_pos,
                    (-direction[0], -direction[1]),
                    contours,
                    dwpose_kp,
                    depth_map,
                    enabled_bones,
                    img_size,
                    set()
                )
            
            # 3. Выбираем лучший вариант
            best_chain = self._select_best_chain(
                chain_from_hip,
                chain_from_foot,
                hip_pos,
                torso
            )
            
            if best_chain:
                leg_chains.append(best_chain)
                all_bones.update(best_chain.bones)
        
        all_chains = arm_chains + leg_chains
        
        debug_print(f"Bidirectional search found {len(all_chains)} chains with {len(all_bones)} bones")
        debug_print(f"  Arms: {len(arm_chains)}, Legs: {len(leg_chains)}")
        
        return all_bones, all_chains, torso, self.body_tilt
    
    def _search_chain(
        self,
        chain_name: str,
        bone_sequence: List[str],
        start_pos: Tuple[float, float],
        initial_direction: Tuple[float, float],
        contours: List[Contour],
        dwpose_kp: Dict[str, Tuple],
        depth_map: Optional[np.ndarray],
        enabled_bones: Set[str],
        img_size: Tuple[int, int],
        used_contours: Set[int]
    ) -> Optional[LimbChain]:
        """Поиск цепочки от начальной точки"""
        
        bones = {}
        current_pos = start_pos
        current_dir = initial_direction
        
        for bone_name in bone_sequence:
            if bone_name not in enabled_bones:
                continue
            
            result = self._search_bone(
                bone_name,
                current_pos,
                current_dir,
                contours,
                dwpose_kp,
                depth_map,
                img_size,
                used_contours
            )
            
            if result:
                bones[bone_name] = result
                
                if result.contour:
                    used_contours.add(result.contour.idx)
                
                # Обновляем направление
                dx = result.position[0] - current_pos[0]
                dy = result.position[1] - current_pos[1]
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0.01:
                    current_dir = (dx/length, dy/length)
                
                current_pos = result.position
        
        if not bones:
            return None
        
        avg_conf = sum(b.confidence for b in bones.values()) / len(bones)
        
        return LimbChain(
            chain_name=chain_name,
            bones=bones,
            start_pos=start_pos,
            confidence=avg_conf
        )
    
    def _search_chain_reverse(
        self,
        chain_name: str,
        bone_sequence: List[str],
        start_pos: Tuple[float, float],
        initial_direction: Tuple[float, float],
        contours: List[Contour],
        dwpose_kp: Dict[str, Tuple],
        depth_map: Optional[np.ndarray],
        enabled_bones: Set[str],
        img_size: Tuple[int, int],
        used_contours: Set[int]
    ) -> Optional[LimbChain]:
        """Поиск цепочки в обратном направлении (от конечности к торсу)"""
        
        # Аналогично _search_chain, но с обратным направлением
        return self._search_chain(
            chain_name,
            bone_sequence,
            start_pos,
            initial_direction,
            contours,
            dwpose_kp,
            depth_map,
            enabled_bones,
            img_size,
            used_contours
        )
    
    def _select_best_chain(
        self,
        chain1: Optional[LimbChain],
        chain2: Optional[LimbChain],
        anchor_pos: Tuple[float, float],
        torso: TrapezoidTorso
    ) -> Optional[LimbChain]:
        """
        Выбирает лучшую цепочку из двух вариантов
        Критерии:
        1. Больше найденных костей
        2. Выше средняя уверенность
        3. Лучшее соединение с торсом
        """
        
        if chain1 is None and chain2 is None:
            return None
        if chain1 is None:
            return chain2
        if chain2 is None:
            return chain1
        
        # Количество костей
        bones1 = len(chain1.bones)
        bones2 = len(chain2.bones)
        
        if bones1 != bones2:
            return chain1 if bones1 > bones2 else chain2
        
        # Уверенность
        if abs(chain1.confidence - chain2.confidence) > 0.1:
            return chain1 if chain1.confidence > chain2.confidence else chain2
        
        # Соединение с торсом (расстояние первой кости от anchor)
        first_bone1 = next(iter(chain1.bones.values()))
        first_bone2 = next(iter(chain2.bones.values()))
        
        dist1 = math.sqrt(
            (first_bone1.position[0] - anchor_pos[0])**2 +
            (first_bone1.position[1] - anchor_pos[1])**2
        )
        dist2 = math.sqrt(
            (first_bone2.position[0] - anchor_pos[0])**2 +
            (first_bone2.position[1] - anchor_pos[1])**2
        )
        
        return chain1 if dist1 < dist2 else chain2
    
    def _search_bone(
        self,
        bone_name: str,
        parent_pos: Tuple[float, float],
        direction: Tuple[float, float],
        contours: List[Contour],
        dwpose_kp: Dict[str, Tuple],
        depth_map: Optional[np.ndarray],
        img_size: Tuple[int, int],
        used_contours: Set[int]
    ) -> Optional[BoneResult]:
        """Поиск одной кости"""
        
        h, w = img_size
        
        # Ожидаемый размер
        expected_size = self.head_props.get_expected_size(bone_name)
        max_distance = self.head_props.head_size * 2.5
        
        # Нормализуем направление
        dir_len = math.sqrt(direction[0]**2 + direction[1]**2)
        if dir_len > 0:
            dir_norm = (direction[0]/dir_len, direction[1]/dir_len)
        else:
            dir_norm = (0, 1)
        
        # Поиск кандидатов
        candidates = []
        for cnt in contours:
            if cnt.idx in used_contours:
                continue
            
            # Расстояние
            dist = math.sqrt(
                (cnt.center[0] - parent_pos[0])**2 +
                (cnt.center[1] - parent_pos[1])**2
            )
            
            if dist > max_distance:
                continue
            
            # Размер
            cnt_size = max(cnt.size)
            if not (expected_size[0] <= cnt_size <= expected_size[1]):
                continue
            
            # Направление
            dx = cnt.center[0] - parent_pos[0]
            dy = cnt.center[1] - parent_pos[1]
            if dist > 0.01:
                dot = (dx * dir_norm[0] + dy * dir_norm[1]) / dist
            else:
                dot = 0
            
            # Скоринг с учетом наклона тела
            distance_score = max(0, 1.0 - dist / max_distance) * 0.3
            direction_score = max(0, dot) * 0.4
            size_score = 0.2
            
            # Бонус за глубину (должна быть схожа с родителем)
            depth_bonus = 0.0
            if depth_map is not None and self.body_tilt:
                parent_depth = self._get_depth(parent_pos[0], parent_pos[1], depth_map, w, h)
                cnt_depth = cnt.depth
                
                # С учетом наклона
                if "leg" in bone_name.lower() or "Thigh" in bone_name or "Calf" in bone_name or "Foot" in bone_name:
                    # Ноги - проверяем ожидаемую глубину
                    if self.body_tilt.direction == "forward":
                        # Наклон вперед - ноги должны быть светлее (ближе)
                        if cnt_depth > parent_depth:
                            depth_bonus = 0.1
                    elif self.body_tilt.direction == "backward":
                        # Наклон назад - ноги темнее
                        if cnt_depth < parent_depth:
                            depth_bonus = 0.1
                else:
                    # Руки - глубина близка к родителю
                    depth_diff = abs(cnt_depth - parent_depth)
                    if depth_diff < 0.15:
                        depth_bonus = 0.1
            
            score = distance_score + direction_score + size_score + depth_bonus
            
            if score > 0.3:
                cnt.search_score = score
                candidates.append(cnt)
        
        # Лучший кандидат
        if candidates:
            best = max(candidates, key=lambda c: c.search_score)
            best.assigned_bone = bone_name
            
            return BoneResult(
                bone_name=bone_name,
                position=best.center,
                confidence=best.search_score,
                depth=best.depth,
                contour=best
            )
        
        # Fallback - DWPose
        if bone_name in dwpose_kp and dwpose_kp[bone_name][2] > 0.3:
            kp = dwpose_kp[bone_name]
            return BoneResult(
                bone_name=bone_name,
                position=(kp[0], kp[1]),
                confidence=kp[2] * 0.6,
                depth=self._get_depth(kp[0], kp[1], depth_map, w, h),
                contour=None
            )
        
        return None
    
    def _get_depth(self, x, y, dm, w, h):
        if dm is None:
            return 0.5
        dh, dw = dm.shape[:2]
        return float(dm[max(0, min(dh-1, int(y*dh))), max(0, min(dw-1, int(x*dw)))])


# =============================================================================
# FINGER GENERATOR (расширенный)
# =============================================================================

class FingerGenerator:
    @staticmethod
    def generate_from_hand(
        hand_pos: Tuple[float, float, float, float],
        forearm_pos: Optional[Tuple[float, float, float, float]],
        side: str,
        enabled_fingers: Set[str]
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """Генерирует пальцы руки"""
        result = {}
        hx, hy, conf, hz = hand_pos
        
        # Направление от предплечья к кисти
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
        
        # Перпендикуляр для растопыривания пальцев
        px, py = -dy, dx
        phalanx_len = 0.02
        
        # Углы для каждого пальца
        finger_angles = {
            "Thumb": -0.6 if side == "R" else 0.6,
            "Index": -0.3 if side == "R" else 0.3,
            "Middle": 0,
            "Ring": 0.3 if side == "R" else -0.3,
            "Pinky": 0.6 if side == "R" else -0.6,
        }
        
        for finger in FINGER_NAMES:
            angle_offset = finger_angles.get(finger, 0)
            
            # Направление пальца
            fx = dx + px * angle_offset
            fy = dy + py * angle_offset
            flen = math.sqrt(fx*fx + fy*fy)
            if flen > 0:
                fx, fy = fx/flen, fy/flen
            
            # Начальная точка (от кисти)
            curr_x, curr_y = hx, hy
            
            # Фаланги
            for phalanx in range(1, PHALANX_COUNT + 1):
                bone_name = f"{finger}_{phalanx}_{side}"
                if bone_name not in enabled_fingers:
                    continue
                
                curr_x += fx * phalanx_len
                curr_y += fy * phalanx_len
                
                # Уверенность убывает с каждой фалангой
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
        enabled_toes: Set[str]
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """Генерирует пальцы ног"""
        result = {}
        fx, fy, conf, fz = foot_pos
        
        # Направление пальцев (вперёд от стопы)
        dx, dy = 0, 0.03  # вниз по изображению
        
        # Перпендикуляр
        px, py = -dy, dx
        
        phalanx_len = 0.015
        
        # Позиции пальцев ног
        toe_offsets = {
            "BigToe": -0.015 if side == "L" else 0.015,
            "IndexToe": -0.01 if side == "L" else 0.01,
            "MiddleToe": 0,
            "RingToe": 0.01 if side == "L" else -0.01,
            "PinkyToe": 0.015 if side == "L" else -0.015,
        }
        
        for toe, offset in toe_offsets.items():
            # Стартовая позиция пальца
            start_x = fx + px * offset
            start_y = fy
            
            curr_x, curr_y = start_x, start_y
            
            for phalanx in range(1, TOE_PHALANX_COUNT + 1):
                bone_name = f"{toe}_{phalanx}_{side}"
                if bone_name not in enabled_toes:
                    continue
                
                curr_x += dx * phalanx_len
                curr_y += dy * phalanx_len
                
                result[bone_name] = (
                    max(0, min(1, curr_x)),
                    max(0, min(1, curr_y)),
                    conf * (0.7 ** phalanx),
                    fz
                )
        
        return result


class SkeletonManager:
    """Менеджер включённых костей и соединений"""
    
    def __init__(self, settings: Dict[str, bool]):
        self.settings = settings
        self._enabled_bones = None
        self._connections = None
    
    @property
    def enabled_bones(self) -> Set[str]:
        if self._enabled_bones is None:
            self._enabled_bones = self._compute_enabled_bones()
        return self._enabled_bones
    
    def _compute_enabled_bones(self) -> Set[str]:
        enabled = set()
        
        for gk, g in UI_GROUPS.items():
            en = self.settings.get(gk, g.default == "enable")
            
            if g.parent_group and g.parent_group in UI_GROUPS:
                parent = UI_GROUPS[g.parent_group]
                en = en and self.settings.get(g.parent_group, parent.default == "enable")
            
            if en:
                enabled.update(g.bones)
        
        # Обязательные кости
        for name, bone in BONE_DEFINITIONS.items():
            if bone.required:
                enabled.add(name)
        
        return self._filter_by_parents(enabled)
    
    def _filter_by_parents(self, enabled: Set[str]) -> Set[str]:
        result = set()
        for bn in enabled:
            if self._has_parent(bn, enabled):
                result.add(bn)
        return result
    
    def _has_parent(self, bn: str, enabled: Set[str]) -> bool:
        bone = BONE_DEFINITIONS.get(bn)
        if not bone:
            return False
        if not bone.parent:
            return True
        if bone.parent == "Pelvis" and "Pelvis" not in enabled:
            return bn in ["Thigh_L", "Thigh_R"] and ("Neck" in enabled or "Root" in enabled)
        if bone.parent not in enabled:
            return False
        return self._has_parent(bone.parent, enabled)
    
    def get_connections(self):
        if self._connections is not None:
            return self._connections
        
        conns = []
        pelvis_en = "Pelvis" in self.enabled_bones
        
        for bn in self.enabled_bones:
            bone = BONE_DEFINITIONS.get(bn)
            if bone and bone.parent and bone.parent in self.enabled_bones:
                conns.append((bone.parent, bn))
        
        if not pelvis_en:
            for c in OPENPOSE_FALLBACK:
                if c[0] in self.enabled_bones and c[1] in self.enabled_bones:
                    conns.append(c)
        
        self._connections = conns
        return conns
    
    def is_enabled(self, bn):
        return bn in self.enabled_bones

# =============================================================================
# =============================================================================
# SKELETON RECONSTRUCTOR (финальная реконструкция)
# =============================================================================

class SkeletonReconstructor:
    def __init__(self, config: RefinementConfig):
        self.config = config
        self.finger_gen = FingerGenerator()
    
    def reconstruct_from_bones(
        self,
        bones: Dict[str, BoneResult],
        enabled_bones: Set[str],
        img_size: Tuple[int, int]
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """Финальная реконструкция скелета с пальцами"""
        
        skeleton = {}
        
        # 1. Основные кости
        for bone_name, bone in bones.items():
            if bone_name in enabled_bones:
                skeleton[bone_name] = (
                    bone.position[0],
                    bone.position[1],
                    bone.confidence,
                    bone.depth
                )
        
        debug_print(f"Reconstructing skeleton: {len(skeleton)} base bones")
        
        # 2. Пальцы рук
        fingers_added = 0
        for side in ["L", "R"]:
            hand_key = f"Hand_{side}"
            forearm_key = f"Forearm_{side}"
            
            if hand_key in skeleton:
                enabled_fingers = {
                    f"{f}_{p}_{side}"
                    for f in FINGER_NAMES
                    for p in range(1, PHALANX_COUNT + 1)
                    if f"{f}_{p}_{side}" in enabled_bones
                }
                
                if enabled_fingers:
                    # Проверяем, сколько уже есть
                    existing = sum(1 for f in enabled_fingers if f in skeleton)
                    
                    # Генерируем только если меньше половины
                    if existing < len(enabled_fingers) * 0.5:
                        fingers = self.finger_gen.generate_from_hand(
                            skeleton[hand_key],
                            skeleton.get(forearm_key),
                            side,
                            enabled_fingers
                        )
                        
                        for k, v in fingers.items():
                            if k not in skeleton:
                                skeleton[k] = v
                                fingers_added += 1
        
        if fingers_added > 0:
            debug_print(f"  Added {fingers_added} finger bones")
        
        # 3. Пальцы ног (если включены)
        toes_added = 0
        for side in ["L", "R"]:
            toe_base_key = f"Toe_{side}"
            foot_key = f"Foot_{side}"
            
            if toe_base_key in skeleton or foot_key in skeleton:
                enabled_toes = {
                    f"{t}_{p}_{side}"
                    for t in TOE_NAMES
                    for p in range(1, TOE_PHALANX_COUNT + 1)
                    if f"{t}_{p}_{side}" in enabled_bones
                }
                
                if enabled_toes:
                    base_pos = skeleton.get(toe_base_key, skeleton.get(foot_key))
                    if base_pos:
                        toes = self.finger_gen.generate_toes(
                            base_pos,
                            side,
                            enabled_toes
                        )
                        
                        for k, v in toes.items():
                            if k not in skeleton:
                                skeleton[k] = v
                                toes_added += 1
        
        if toes_added > 0:
            debug_print(f"  Added {toes_added} toe bones")
        
        debug_print(f"✓ Final skeleton: {len(skeleton)} total bones")
        return skeleton

# =============================================================================
# SKELETON REFINER (обновлённый)
# =============================================================================

class SkeletonRefiner:
    def __init__(self, config: RefinementConfig):
        self.config = config
        self.depth_proc = DepthProcessor(config.depth_model) if config.use_depth else None
        self.edge_proc = EdgeProcessor(config.edge_model) if config.use_edge else None
        self.contour_finder = ContourFinder(config)
        self.skeleton_builder = MultiChannelSkeletonBuilder()
        self.reconstructor = SkeletonReconstructor(config)
        
        self._original_image = None
        self._body_mask = None
        self._depth_map = None
        self._edge_map = None
        self._normals = None
        self._convexity_map = None
        self._contours = []
        self._bones = {}
        self._metadata = {}
        self._size = (0, 0)
    
    def compute(self, image: np.ndarray, dwpose_kp: Dict, face_kp: List):
        h, w = image.shape[:2]
        self._size = (h, w)
        self._original_image = image.copy()
        
        # Простая маска тела
        self._body_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Глубина
        if self.depth_proc and self.config.use_depth:
            debug_print("Computing depth map...")
            depth = self.depth_proc.process(image)
            if depth is not None:
                self._depth_map = cv2.resize(depth, (w, h)) if depth.shape[:2] != (h, w) else depth
                self._normals = ConvexNormalsHelper.compute_surface_normals(self._depth_map)
                self._convexity_map = ConvexNormalsHelper.compute_convexity(self._depth_map)
                debug_print("✓ Depth, normals, convexity computed")
        
        # Границы
        if self.edge_proc and self.config.use_edge:
            debug_print("Computing edge map...")
            edge = self.edge_proc.process(
                image,
                self.config.edge_threshold_low,
                self.config.edge_threshold_high
            )
            if edge is not None:
                self._edge_map = cv2.resize(edge, (w, h)) if edge.shape[:2] != (h, w) else edge
                debug_print("✓ Edge map computed")
        
        # Контуры
        if self.config.use_contour_analysis and self._edge_map is not None:
            debug_print("Finding contours...")
            self._contours = self.contour_finder.find_contours(
                self._edge_map,
                self._depth_map,
                self._body_mask,
                self._convexity_map,
                (h, w)
            )
    
    def refine(self, dwpose_kp, enabled_bones, img_hw):
    def get_depth_visual(self):
        """Визуализация карты глубины"""
        if self._depth_map is None:
            return None
        
        # Цветная визуализация (горячая карта)
        depth_norm = (self._depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
        
        # Добавляем легенду
        h, w = depth_colored.shape[:2]
        legend_h = 30
        legend = np.zeros((legend_h, w, 3), dtype=np.uint8)
        
        # Градиент для легенды
        for i in range(w):
            val = int(255 * i / w)
            legend[:, i] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0]
        
        # Текст
        cv2.putText(legend, "Far", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(legend, "Near", (w - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        result = np.vstack([depth_colored, legend])
        return result
    
    def get_normals_visual(self):
        """Визуализация нормалей поверхности"""
        if self._normals is None:
            return None
        
        normals_vis = ConvexNormalsHelper.visualize_normals(self._normals)
        
        # Добавляем легенду
        h, w = normals_vis.shape[:2]
        cv2.putText(normals_vis, "Surface Normals (RGB = XYZ)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(normals_vis, "R: Left/Right | G: Up/Down | B: Forward/Back", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return normals_vis
    
    def get_edge_visual(self):
        """Визуализация границ"""
        if self._edge_map is None:
            return None
        return cv2.cvtColor(self._edge_map, cv2.COLOR_GRAY2RGB)
    
    def clear(self):
        """Очистка всех данных"""
        self._original_image = None
        self._body_mask = None
        self._depth_map = None
        self._edge_map = None
        self._normals = None
        self._convexity_map = None
        self._contours = []
        self._bones = {}
        self._metadata = {}
        self._size = (0, 0)
    
    def unload(self):
        """Выгрузка моделей и очистка памяти"""
        self.clear()
        if self.depth_proc:
            self.depth_proc.unload()
        if self.edge_proc:
            self.edge_proc.unload()
            
        """Многоканальное построение скелета"""
        
        debug_print("\n" + "="*60)
        debug_print("MULTI-CHANNEL SKELETON RECONSTRUCTION")
        debug_print("="*60)
        
        # Многоканальный поиск
        self._bones, self._metadata = self.skeleton_builder.build_skeleton(
            self._original_image,
            dwpose_kp,
            [],  # face_kp передаётся в compute
            self._contours,
            self._depth_map,
            self._normals,
            self._body_mask,
            enabled_bones,
            img_hw
        )
        
        # Финальная реконструкция с пальцами
        skeleton = self.reconstructor.reconstruct_from_bones(
            self._bones,
            enabled_bones,
            img_hw
        )
        
        debug_print(f"\n✓ Final skeleton: {len(skeleton)} bones")
        debug_print("="*60 + "\n")
        
        return skeleton
    
    # ... (get_depth_visual, get_normals_visual, get_edge_visual - без изменений)
    
    def get_analysis_visual(self, face_kp: List = None):
    # === МЕТОДЫ ОЧИСТКИ И ВИЗУАЛИЗАЦИИ ===
    
    def get_depth_visual(self):
        if self._depth_map is None:
            return None
        depth_norm = (self._depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
        h, w = depth_colored.shape[:2]
        legend_h = 30
        legend = np.zeros((legend_h, w, 3), dtype=np.uint8)
        for i in range(w):
            val = int(255 * i / w)
            legend[:, i] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0]
        cv2.putText(legend, "Far", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(legend, "Near", (w - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        result = np.vstack([depth_colored, legend])
        return result
    
    def get_normals_visual(self):
        if self._normals is None:
            return None
        normals_vis = ConvexNormalsHelper.visualize_normals(self._normals)
        h, w = normals_vis.shape[:2]
        cv2.putText(normals_vis, "Surface Normals (RGB = XYZ)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(normals_vis, "R: Left/Right | G: Up/Down | B: Forward/Back", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return normals_vis
    
    def get_edge_visual(self):
        if self._edge_map is None:
            return None
        return cv2.cvtColor(self._edge_map, cv2.COLOR_GRAY2RGB)
    
    def clear(self):
        """Очистка всех временных данных"""
        self._original_image = None
        self._body_mask = None
        self._depth_map = None
        self._edge_map = None
        self._normals = None
        self._convexity_map = None
        self._contours = []
        self._bones = {}
        self._metadata = {}
        self._size = (0, 0)
    
    def unload(self):
        """Полная выгрузка с освобождением моделей"""
        self.clear()
        if self.depth_proc:
            self.depth_proc.unload()
        if self.edge_proc:
            self.edge_proc.unload()

        """РАСШИРЕННАЯ визуализация с примитивами и кандидатами"""
        
        if self._size == (0, 0):
            return None
        
        h, w = self._size
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Полупрозрачная подложка
        if self._original_image is not None:
            alpha = 0.15
            vis = (self._original_image * alpha).astype(np.uint8)
        
        # Силуэт
        if self._metadata.get("silhouette") is not None:
            silhouette = self._metadata["silhouette"]
            vis[silhouette > 0] = (40, 40, 80)
        
        # Выпуклые области
        for center, size, region_type in self._metadata.get("convex_regions", []):
            cx, cy = int(center[0] * w), int(center[1] * h)
            radius = int(size * w * 0.5)
            
            color = {
                "head": (255, 200, 200),
                "torso": (200, 255, 200),
                "limb": (200, 200, 255)
            }.get(region_type, (150, 150, 150))
            
            cv2.circle(vis, (cx, cy), radius, color, 2)
            cv2.putText(vis, region_type, (cx + 5, cy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Геометрические примитивы
        for prim in self._metadata.get("primitives", []):
            cx, cy = int(prim.center[0] * w), int(prim.center[1] * h)
            
            color = {
                GeometricPrimitive.ELLIPSE: (255, 255, 0),
                GeometricPrimitive.TRAPEZOID: (0, 255, 255),
                GeometricPrimitive.CONE: (255, 0, 255),
                GeometricPrimitive.CIRCLE: (0, 255, 0),
            }.get(prim.primitive_type, (128, 128, 128))
            
            if prim.contour and prim.contour.cv_ellipse:
                cv2.ellipse(vis, prim.contour.cv_ellipse, color, 2)
            
            cv2.putText(vis, prim.primitive_type.value[:4], (cx - 15, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Кандидаты (все источники)
        all_candidates = self._metadata.get("all_candidates", {})
        for bone_name, candidates in all_candidates.items():
            for candidate in candidates:
                cx, cy = int(candidate.position[0] * w), int(candidate.position[1] * h)
                
                # Цвет по источнику
                source_colors = {
                    "dwpose": (0, 255, 0),
                    "primitive": (255, 0, 255),
                    "silhouette": (0, 255, 255),
                    "estimated": (128, 128, 128),
                }
                
                for src in candidate.source.split("+"):
                    color = source_colors.get(src, (255, 255, 255))
                
                cv2.circle(vis, (cx, cy), 4, color, -1)
                cv2.circle(vis, (cx, cy), 5, (255, 255, 255), 1)
        
        # Финальные кости (крупно)
        for bone_name, bone in self._bones.items():
            bx, by = int(bone.position[0] * w), int(bone.position[1] * h)
            
            cv2.circle(vis, (bx, by), 7, (0, 255, 0), -1)
            cv2.circle(vis, (bx, by), 8, (255, 255, 255), 2)
            
            cv2.putText(vis, bone_name, (bx + 10, by - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Уверенность
            cv2.putText(vis, f"{bone.confidence:.2f}", (bx + 10, by + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 255, 200), 1)
        
        # Соединения
        connections = self.skeleton_builder._get_connections(self._bones, set(self._bones.keys()))
        for b1, b2 in connections:
            if b1 in self._bones and b2 in self._bones:
                p1 = (int(self._bones[b1].position[0] * w), int(self._bones[b1].position[1] * h))
                p2 = (int(self._bones[b2].position[0] * w), int(self._bones[b2].position[1] * h))
                cv2.line(vis, p1, p2, (0, 200, 200), 2)
        
        # Валидационные оценки
        validation = self._metadata.get("validation_scores", {})
        if validation:
            y = 20
            cv2.putText(vis, "Validation Scores:", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
            for bone_name, score in list(validation.items())[:10]:
                color = (0, 255, 0) if score > 0.7 else (255, 255, 0) if score > 0.4 else (255, 0, 0)
                cv2.putText(vis, f"  {bone_name}: {score:.2f}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y += 15
        
        # Легенда
        legend_y = h - 120
        cv2.rectangle(vis, (w - 205, legend_y - 25), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.rectangle(vis, (w - 205, legend_y - 25), (w - 5, h - 5), (100, 100, 100), 2)
        
        cv2.putText(vis, "Sources:", (w - 200, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 18
        
        for source, color in [("DWPose", (0, 255, 0)), ("Primitive", (255, 0, 255)),
                              ("Silhouette", (0, 255, 255)), ("Estimated", (128, 128, 128))]:
            cv2.circle(vis, (w - 190, legend_y), 4, color, -1)
            cv2.putText(vis, source, (w - 175, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            legend_y += 18
        
        return vis

# =============================================================================
# VISUALIZER
# =============================================================================

class SkeletonVisualizer:
    @staticmethod
    def draw(canvas, kp, conns, face_kp=None, radius=6, thick=3, min_conf=0.05):
        """Отрисовка финального скелета"""
        res = canvas.copy()
        h, w = res.shape[:2]
        
        # Лицо
        if face_kp:
            for fp in face_kp:
                if len(fp) >= 3 and fp[2] > 0.3:
                    fx, fy = int(fp[0] * w), int(fp[1] * h)
                    cv2.circle(res, (fx, fy), 1, (200, 200, 255), -1)
        
        # Точки костей
        pts = {}
        for bn, k in kp.items():
            if len(k) >= 3 and k[2] >= min_conf and 0 <= k[0] <= 1 and 0 <= k[1] <= 1:
                pts[bn] = (int(k[0] * w), int(k[1] * h), k[2])
        
        # Линии связей
        for b1, b2 in conns:
            if b1 in pts and b2 in pts:
                p1, p2 = pts[b1], pts[b2]
                bd = BONE_DEFINITIONS.get(b2)
                col = BONE_COLORS.get(bd.category if bd else BoneCategory.CORE, (255, 255, 255))
                cv2.line(res, (p1[0], p1[1]), (p2[0], p2[1]), col, thick, cv2.LINE_AA)
        
        # Точки костей
        for bn, pt in pts.items():
            bd = BONE_DEFINITIONS.get(bn)
            col = BONE_COLORS.get(bd.category if bd else BoneCategory.CORE, (255, 255, 255))
            cv2.circle(res, (pt[0], pt[1]), radius, col, -1, cv2.LINE_AA)
            cv2.circle(res, (pt[0], pt[1]), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        return res



# =============================================================================
# MAIN NODE
# =============================================================================

class DWPoseExtended:
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("pose_image", "keypoints", "depth_map", "normals_map", "edge_map", "analysis")
    FUNCTION = "process"
    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = OrderedDict()
        inputs["image"] = ("IMAGE",)
        inputs["resolution"] = INPUT.RESOLUTION()
        inputs["bbox_detector"] = INPUT.COMBO(
            ["yolox_l.onnx", "yolox_l.torchscript.pt", "yolo_nas_l_fp16.onnx", "None"],
            default="yolox_l.onnx"
        )
        inputs["pose_estimator"] = INPUT.COMBO(
            ["dw-ll_ucoco_384_bs5.torchscript.pt", "dw-ll_ucoco_384.onnx", "dw-ll_ucoco.onnx"],
            default="dw-ll_ucoco_384_bs5.torchscript.pt"
        )
        inputs["detect_body"] = INPUT.COMBO(["enable", "disable"], default="enable")
        inputs["detect_face"] = INPUT.COMBO(["enable", "disable"], default="enable")
        inputs["detect_hands"] = INPUT.COMBO(["enable", "disable"], default="enable")
        inputs["use_depth"] = INPUT.COMBO(["enable", "disable"], default="enable")
        inputs["use_edge"] = INPUT.COMBO(["enable", "disable"], default="enable")
        inputs["use_contour_analysis"] = INPUT.COMBO(["enable", "disable"], default="enable")
        inputs["depth_model"] = INPUT.COMBO(list(DEPTH_MODELS.keys()), default="midas_v21_small")
        inputs["edge_model"] = INPUT.COMBO(list(EDGE_MODELS.keys()), default="canny")
        
        for gk, g in sorted(UI_GROUPS.items(), key=lambda x: x[1].order):
            inputs[gk] = INPUT.COMBO(["enable", "disable"], default=g.default)
        
        return define_preprocessor_inputs(**inputs)
    
    def __init__(self):
        self.openpose_dicts = []
        self.skel_mgr = None
        self.refiner = None
    
    def _repo(self, fn):
        if fn in MODEL_REPOS:
            return MODEL_REPOS[fn]
        if fn == "None":
            return DWPOSE_MODEL_NAME
        if "yolox" in fn:
            return "hr16/yolox-onnx" if ".torchscript" in fn else DWPOSE_MODEL_NAME
        if "yolo_nas" in fn:
            return "hr16/yolo-nas-fp16"
        return DWPOSE_MODEL_NAME
    
    def _settings(self, **kw):
        return {k: kw.get(k, g.default) == "enable" for k, g in UI_GROUPS.items()}
    
    def _load(self, bbox, pose):
        """Загрузка DWPose детектора"""
        try:
            from custom_controlnet_aux.dwpose import DwposeDetector
            
            yr, pr = self._repo(bbox), self._repo(pose)
            dev = _get_device()
            
            debug_print(f"Loading DWPose: bbox={yr}, pose={pr}, device={dev}")
            
            try:
                detector = DwposeDetector.from_pretrained(
                    pr, yr,
                    det_filename=(None if bbox == "None" else bbox),
                    pose_filename=pose,
                    torchscript_device=dev
                )
                debug_print("✓ DWPose loaded successfully")
                return detector
                
            except Exception as e:
                if "cuda" in str(e).lower() or "hip" in str(e).lower():
                    debug_print(f"GPU error: {e}, trying CPU")
                    _set_cpu_fallback(True)
                    detector = DwposeDetector.from_pretrained(
                        pr, yr,
                        det_filename=(None if bbox == "None" else bbox),
                        pose_filename=pose,
                        torchscript_device=torch.device("cpu")
                    )
                    debug_print("✓ DWPose loaded on CPU")
                    return detector
                else:
                    raise
                    
        except ImportError as e:
            debug_print(f"✗ Failed to import DwposeDetector: {e}")
            raise ImportError(
                "DWPose detector not found. "
                "Please install comfyui_controlnet_aux:\n"
                "cd custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux"
            )
    
    def _parse(self, pd, hw, include_face=True):
        """Парсинг результатов DWPose"""
        kp = {}
        face_kp = []
        h, w = hw
        
        if not pd:
            return kp, face_kp
        
        ch, cw = pd.get("canvas_height", h), pd.get("canvas_width", w)
        
        def norm(x, y, s):
            if x <= 0 or y <= 0 or s < 0.01:
                return None
            xn = x / cw if x > 1.5 else x
            yn = y / ch if y > 1.5 else y
            if not (0 <= xn <= 1 and 0 <= yn <= 1):
                return None
            return (xn, yn, s)
        
        def pts(d):
            if not d:
                return []
            if isinstance(d, np.ndarray):
                d = d.tolist()
            r = []
            if d and isinstance(d[0], (list, tuple, np.ndarray)):
                for p in d:
                    if len(p) >= 2:
                        r.append((float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.8))
            elif d and isinstance(d[0], (int, float)):
                for i in range(0, len(d) - 2, 3):
                    r.append((float(d[i]), float(d[i+1]), float(d[i+2])))
            return r
        
        for person in pd.get("people", []):
            if not isinstance(person, dict):
                continue
            
            bd = person.get("pose_keypoints_2d") or person.get("body") or person.get("keypoints")
            if bd:
                ps = pts(bd)
                hps = []
                for i, (x, y, s) in enumerate(ps):
                    if i <= 4:
                        n = norm(x, y, s)
                        if n:
                            hps.append(n)
                    elif i in COCO_TO_SKELETON:
                        n = norm(x, y, s)
                        if n:
                            kp[COCO_TO_SKELETON[i]] = n
                if hps:
                    kp["Head"] = (
                        sum(p[0] for p in hps) / len(hps),
                        sum(p[1] for p in hps) / len(hps),
                        sum(p[2] for p in hps) / len(hps)
                    )
            
            if include_face:
                face_data = person.get("face_keypoints_2d") or person.get("face")
                if face_data:
                    for x, y, s in pts(face_data):
                        n = norm(x, y, s)
                        if n:
                            face_kp.append(n)
            
            for k, sd in [("hand_left_keypoints_2d", "L"), ("hand_right_keypoints_2d", "R")]:
                if k in person and person[k]:
                    for i, (x, y, s) in enumerate(pts(person[k])):
                        n = norm(x, y, s)
                        if n and i in HAND_KEYPOINT_MAP:
                            f, p = HAND_KEYPOINT_MAP[i]
                            if p > 0:
                                kp[f"{f}_{p}_{sd}"] = n
        
        return kp, face_kp
    
    def process(
        self, image,
        resolution=512,
        bbox_detector="yolox_l.onnx",
        pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
        detect_body="enable",
        detect_face="enable",
        detect_hands="enable",
        use_depth="enable",
        use_edge="enable",
        use_contour_analysis="enable",
        depth_model="midas_v21_small",
        edge_model="canny",
        **kw
    ):
        debug_print("\n" + "="*80)
        debug_print("DWPose Extended - Multi-Channel 3D Skeleton Reconstruction")
        debug_print("  ✓ Perspective analysis with priority search")
        debug_print("  ✓ Geometric primitives (trapezoid torso, cones, circles)")
        debug_print("  ✓ Silhouette convex regions detection")
        debug_print("  ✓ Anatomical validation of connections")
        debug_print("  ✓ Multi-source fusion (DWPose + Primitives + Silhouette)")
        debug_print("="*80 + "\n")
        
        self.skel_mgr = SkeletonManager(self._settings(**kw))
        
        cfg = RefinementConfig(
            use_depth=(use_depth == "enable"),
            use_edge=(use_edge == "enable"),
            use_contour_analysis=(use_contour_analysis == "enable"),
            depth_model=depth_model,
            edge_model=edge_model
        )
        self.refiner = SkeletonRefiner(cfg)
        
        model = self._load(bbox_detector, pose_estimator)
        
        inc_body = detect_body == "enable"
        inc_face = detect_face == "enable"
        inc_hand = detect_hands == "enable"
        
        self.openpose_dicts = []
        all_pose, all_depth, all_normals, all_edge, all_analysis = [], [], [], [], []
        
        def proc(img, **k):
            h, w = img.shape[:2]
            debug_print(f"\nProcessing image: {w}x{h}")
            
            _, pd = model(img, **k)
            dwkp, face_kp = self._parse(pd, (h, w), include_face=inc_face)
            
            debug_print(f"  DWPose keypoints: {len(dwkp)}")
            debug_print(f"  Face keypoints: {len(face_kp)}")
            
            self.refiner.clear()
            self.refiner.compute(img, dwkp, face_kp)
            
            skel = self.refiner.refine(dwkp, self.skel_mgr.enabled_bones, (h, w))
            skel = {k: v for k, v in skel.items() if self.skel_mgr.is_enabled(k)}
            
            debug_print(f"\n✓ Final skeleton: {len(skel)} bones")
            debug_print(f"  Core: {[k for k in skel.keys() if 'Head' in k or 'Neck' in k or 'Spine' in k or 'Pelvis' in k or 'Root' in k]}")
            debug_print(f"  Arms: {[k for k in skel.keys() if 'Shoulder' in k or 'Forearm' in k or 'Hand' in k or 'Clavicle' in k]}")
            debug_print(f"  Legs: {[k for k in skel.keys() if 'Thigh' in k or 'Calf' in k or 'Foot' in k]}")
            
            # Визуализация скелета
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            pv = SkeletonVisualizer.draw(
                canvas, skel, self.skel_mgr.get_connections(),
                face_kp=face_kp if inc_face else None,
                radius=max(4, min(h, w) // 100),
                thick=max(2, min(h, w) // 120)
            )
            all_pose.append(pv)
            
            # Карта глубины
            dv = self.refiner.get_depth_visual()
            all_depth.append(dv if dv is not None else np.zeros((h, w, 3), dtype=np.uint8))
            
            # Нормали
            nv = self.refiner.get_normals_visual()
            all_normals.append(nv if nv is not None else np.zeros((h, w, 3), dtype=np.uint8))
            
            # Границы
            ev = self.refiner.get_edge_visual()
            all_edge.append(ev if ev is not None else np.zeros((h, w, 3), dtype=np.uint8))
            
            # Анализ
            av = self.refiner.get_analysis_visual(face_kp=face_kp if inc_face else None)
            all_analysis.append(av if av is not None else np.zeros((h, w, 3), dtype=np.uint8))
            
            # JSON данные
            perspective = self.refiner._metadata.get("perspective")
            validation = self.refiner._metadata.get("validation_scores", {})
            
            self.openpose_dicts.append({
                "skeleton": {
                    k: {"x": v[0], "y": v[1], "confidence": v[2], "z": v[3] if len(v) > 3 else 0}
                    for k, v in skel.items()
                },
                "face_keypoints": [{"x": f[0], "y": f[1], "confidence": f[2]} for f in face_kp],
                "connections": self.skel_mgr.get_connections(),
                "perspective": {
                    "tilt_angle": perspective.tilt_angle if perspective else 0,
                    "near_parts": perspective.near_body_parts if perspective else [],
                    "far_parts": perspective.far_body_parts if perspective else [],
                    "is_full_body": perspective.is_full_body if perspective else False,
                } if perspective else None,
                "validation_scores": validation,
                "bone_sources": {
                    bone_name: self.refiner._bones[bone_name].source
                    for bone_name in self.refiner._bones.keys()
                } if hasattr(self.refiner, '_bones') else {},
                "primitives_count": len(self.refiner._metadata.get("primitives", [])),
                "convex_regions_count": len(self.refiner._metadata.get("convex_regions", [])),
                "image_size": {"width": w, "height": h}
            })
            
            return pv
        
        common_annotator_call(
            proc, image,
            include_hand=inc_hand,
            include_face=inc_face,
            include_body=inc_body,
            image_and_json=True,
            resolution=resolution
        )
        
        del model
        self.refiner.unload()
        
        def tens(imgs):
            return torch.from_numpy(np.stack(imgs)).float() / 255.0 if imgs else torch.zeros((1, 64, 64, 3))
        
        debug_print("\n" + "="*80)
        debug_print("✓ Processing complete!")
        debug_print("="*80 + "\n")
        
        return {
            'ui': {'pose_json': [json.dumps(self.openpose_dicts, indent=2)]},
            'result': (
                tens(all_pose),
                self.openpose_dicts,
                tens(all_depth),
                tens(all_normals),
                tens(all_edge),
                tens(all_analysis)
            )
        }


NODE_CLASS_MAPPINGS = {"DWPoseExtended": DWPoseExtended}
NODE_DISPLAY_NAME_MAPPINGS = {"DWPoseExtended": "DWPose Extended (Multi-Channel 3D)"}