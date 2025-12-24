"""
DWPose Extended - Данные и константы
Этот файл содержит только данные и не должен часто изменяться
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import torch
import numpy as np
from einops import rearrange

# =============================================================================
# КОНСТАНТЫ МОДЕЛЕЙ
# =============================================================================

DWPOSE_MODEL_NAME = "yzd-v/DWPose"

MODEL_REPOS = {
    "yolox_l.onnx": "yzd-v/DWPose",
    "yolox_l.torchscript.pt": "hr16/yolox-onnx",
    "yolo_nas_l_fp16.onnx": "hr16/yolo-nas-fp16",
    "yolo_nas_m_fp16.onnx": "hr16/yolo-nas-fp16",
    "yolo_nas_s_fp16.onnx": "hr16/yolo-nas-fp16",
    "dw-ll_ucoco_384.onnx": "yzd-v/DWPose",
    "dw-ll_ucoco.onnx": "hr16/UnJIT-DWPose",
    "dw-ll_ucoco_384_bs5.torchscript.pt": "hr16/DWPose-TorchScript-BatchSize5",
}

DEPTH_MODELS = {
    "midas_v21_small": "Intel/dpt-hybrid-midas",
    "midas_v21": "Intel/dpt-large",
    "depth_anything": "LiheYoung/depth-anything-small-hf",
}

EDGE_MODELS = {
    "canny": None,
    "hed": "lllyasviel/Annotators",
    "pidinet": "lllyasviel/Annotators",
}

GPU_PROVIDERS = [
    "DmlExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "CPUExecutionProvider",
]

# =============================================================================
# ENUM ТИПЫ
# =============================================================================

class BoneCategory(Enum):
    CORE = "core"
    HEAD = "head"
    ARM_L = "arm_left"
    ARM_R = "arm_right"
    HAND_L = "hand_left"
    HAND_R = "hand_right"
    LEG_L = "leg_left"
    LEG_R = "leg_right"
    FOOT_L = "foot_left"
    FOOT_R = "foot_right"


class Side(Enum):
    LEFT = "L"
    RIGHT = "R"
    CENTER = "C"


class FaceDirection(Enum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


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


class GeometricPrimitive(Enum):
    ELLIPSE = "ellipse"
    TRAPEZOID = "trapezoid"
    RECTANGLE = "rectangle"
    CONE = "cone"
    CIRCLE = "circle"
    COMPOSITE = "composite"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class BoneDefinition:
    name: str
    parent: Optional[str]
    category: BoneCategory
    side: Side = Side.CENTER
    required: bool = True
    default_enabled: bool = True
    ui_group: str = ""
    ui_order: int = 0


@dataclass
class UIGroup:
    key: str
    display_name: str
    parent_group: Optional[str] = None
    bones: List[str] = field(default_factory=list)
    default: str = "enable"
    order: int = 0


@dataclass
class BodyPartProportions:
    bone_name: str
    size_relative_to_head: Tuple[float, float]
    offset_from_head: Tuple[float, float]
    search_area_size: Tuple[float, float]
    aspect_ratio: Tuple[float, float]
    priority: int = 50


@dataclass
class RefinementConfig:
    use_depth: bool = True
    use_edge: bool = True
    use_contour_analysis: bool = True
    depth_weight: float = 0.3
    edge_weight: float = 0.4
    contour_weight: float = 0.5
    edge_threshold_low: int = 30
    edge_threshold_high: int = 100
    depth_model: str = "midas_v21_small"
    edge_model: str = "canny"
    contour_search_radius: int = 25
    min_contour_confidence: float = 0.3
    min_point_confidence: float = 0.1
    min_contour_area: int = 100
    ellipse_fit_quality: float = 0.6
    convexity_threshold: float = 0.3
    depth_tolerance: float = 0.2


# =============================================================================
# НАЗВАНИЯ
# =============================================================================

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_DISPLAY_NAMES = {
    "Thumb": "Большой", "Index": "Указательный", "Middle": "Средний",
    "Ring": "Безымянный", "Pinky": "Мизинец"
}

TOE_NAMES = ["BigToe", "IndexToe", "MiddleToe", "RingToe", "PinkyToe"]
TOE_DISPLAY_NAMES = {
    "BigToe": "Большой", "IndexToe": "Второй", "MiddleToe": "Средний",
    "RingToe": "Четвёртый", "PinkyToe": "Мизинец"
}

PHALANX_COUNT = 3
TOE_PHALANX_COUNT = 2

# =============================================================================
# КОСТИ
# =============================================================================

BONE_DEFINITIONS: Dict[str, BoneDefinition] = {
    "Root": BoneDefinition("Root", None, BoneCategory.CORE, required=True, ui_group="core", ui_order=0),
    "Pelvis": BoneDefinition("Pelvis", "Root", BoneCategory.CORE, required=False, ui_group="core", ui_order=1),
    "Spine_1": BoneDefinition("Spine_1", "Pelvis", BoneCategory.CORE, required=False, ui_group="core", ui_order=2),
    "Spine_2": BoneDefinition("Spine_2", "Spine_1", BoneCategory.CORE, required=False, ui_group="core", ui_order=3),
    "Spine_3": BoneDefinition("Spine_3", "Spine_2", BoneCategory.CORE, required=False, ui_group="core", ui_order=4),
    "Neck": BoneDefinition("Neck", "Spine_3", BoneCategory.HEAD, required=True, ui_group="head", ui_order=0),
    "Head": BoneDefinition("Head", "Neck", BoneCategory.HEAD, required=True, ui_group="head", ui_order=1),
    
    "Clavicle_L": BoneDefinition("Clavicle_L", "Spine_3", BoneCategory.ARM_L, Side.LEFT, True, True, "arm_left", 0),
    "Shoulder_L": BoneDefinition("Shoulder_L", "Clavicle_L", BoneCategory.ARM_L, Side.LEFT, True, True, "arm_left", 1),
    "Forearm_L": BoneDefinition("Forearm_L", "Shoulder_L", BoneCategory.ARM_L, Side.LEFT, True, True, "arm_left", 2),
    "Hand_L": BoneDefinition("Hand_L", "Forearm_L", BoneCategory.HAND_L, Side.LEFT, True, True, "hand_left", 0),
    
    "Clavicle_R": BoneDefinition("Clavicle_R", "Spine_3", BoneCategory.ARM_R, Side.RIGHT, True, True, "arm_right", 0),
    "Shoulder_R": BoneDefinition("Shoulder_R", "Clavicle_R", BoneCategory.ARM_R, Side.RIGHT, True, True, "arm_right", 1),
    "Forearm_R": BoneDefinition("Forearm_R", "Shoulder_R", BoneCategory.ARM_R, Side.RIGHT, True, True, "arm_right", 2),
    "Hand_R": BoneDefinition("Hand_R", "Forearm_R", BoneCategory.HAND_R, Side.RIGHT, True, True, "hand_right", 0),
    
    "Thigh_L": BoneDefinition("Thigh_L", "Pelvis", BoneCategory.LEG_L, Side.LEFT, True, True, "leg_left", 0),
    "Calf_L": BoneDefinition("Calf_L", "Thigh_L", BoneCategory.LEG_L, Side.LEFT, True, True, "leg_left", 1),
    "Foot_L": BoneDefinition("Foot_L", "Calf_L", BoneCategory.FOOT_L, Side.LEFT, True, True, "foot_left", 0),
    "Toe_L": BoneDefinition("Toe_L", "Foot_L", BoneCategory.FOOT_L, Side.LEFT, False, False, "foot_left", 1),
    
    "Thigh_R": BoneDefinition("Thigh_R", "Pelvis", BoneCategory.LEG_R, Side.RIGHT, True, True, "leg_right", 0),
    "Calf_R": BoneDefinition("Calf_R", "Thigh_R", BoneCategory.LEG_R, Side.RIGHT, True, True, "leg_right", 1),
    "Foot_R": BoneDefinition("Foot_R", "Calf_R", BoneCategory.FOOT_R, Side.RIGHT, True, True, "foot_right", 0),
    "Toe_R": BoneDefinition("Toe_R", "Foot_R", BoneCategory.FOOT_R, Side.RIGHT, False, False, "foot_right", 1),
}

# Генерация пальцев рук и ног
def _generate_finger_bones():
    bones = {}
    for side_enum, side_str, ui_group in [(Side.LEFT, "L", "hand_left"), (Side.RIGHT, "R", "hand_right")]:
        category = BoneCategory.HAND_L if side_str == "L" else BoneCategory.HAND_R
        for finger in FINGER_NAMES:
            for phalanx in range(1, PHALANX_COUNT + 1):
                bone_name = f"{finger}_{phalanx}_{side_str}"
                parent = f"Hand_{side_str}" if phalanx == 1 else f"{finger}_{phalanx - 1}_{side_str}"
                bones[bone_name] = BoneDefinition(
                    bone_name, parent, category, side_enum, False, True,
                    f"{ui_group}_{finger.lower()}", phalanx
                )
    return bones

def _generate_toe_bones():
    bones = {}
    for side_enum, side_str, ui_group in [(Side.LEFT, "L", "foot_left"), (Side.RIGHT, "R", "foot_right")]:
        category = BoneCategory.FOOT_L if side_str == "L" else BoneCategory.FOOT_R
        for toe in TOE_NAMES:
            for phalanx in range(1, TOE_PHALANX_COUNT + 1):
                bone_name = f"{toe}_{phalanx}_{side_str}"
                parent = f"Toe_{side_str}" if phalanx == 1 else f"{toe}_{phalanx - 1}_{side_str}"
                bones[bone_name] = BoneDefinition(
                    bone_name, parent, category, side_enum, False, False,
                    f"{ui_group}_{toe.lower()}", phalanx
                )
    return bones

BONE_DEFINITIONS.update(_generate_finger_bones())
BONE_DEFINITIONS.update(_generate_toe_bones())

# =============================================================================
# ЦЕПОЧКИ
# =============================================================================

LIMB_CHAINS = {
    "arm_L": ["Clavicle_L", "Shoulder_L", "Forearm_L", "Hand_L"],
    "arm_R": ["Clavicle_R", "Shoulder_R", "Forearm_R", "Hand_R"],
    "leg_L": ["Thigh_L", "Calf_L", "Foot_L"],
    "leg_R": ["Thigh_R", "Calf_R", "Foot_R"],
}

# =============================================================================
# UI ГРУППЫ
# =============================================================================

UI_GROUPS: Dict[str, UIGroup] = {
    "body": UIGroup("body", "🦴 Body", bones=["Root", "Pelvis", "Spine_1", "Spine_2", "Spine_3"], order=0),
    "pelvis_spine": UIGroup("pelvis_spine", "├─ Pelvis & Spine", "body", ["Pelvis", "Spine_1", "Spine_2", "Spine_3"], order=1),
    "head": UIGroup("head", "👤 Head", bones=["Neck", "Head"], order=10),
    "arm_left": UIGroup("arm_left", "💪 Left Arm", bones=["Clavicle_L", "Shoulder_L", "Forearm_L"], order=20),
    "hand_left": UIGroup("hand_left", "✋ Left Hand", "arm_left", ["Hand_L"], order=21),
    "arm_right": UIGroup("arm_right", "💪 Right Arm", bones=["Clavicle_R", "Shoulder_R", "Forearm_R"], order=30),
    "hand_right": UIGroup("hand_right", "✋ Right Hand", "arm_right", ["Hand_R"], order=31),
    "leg_left": UIGroup("leg_left", "🦵 Left Leg", bones=["Thigh_L", "Calf_L"], order=40),
    "foot_left": UIGroup("foot_left", "🦶 Left Foot", "leg_left", ["Foot_L", "Toe_L"], order=41),
    "leg_right": UIGroup("leg_right", "🦵 Right Leg", bones=["Thigh_R", "Calf_R"], order=50),
    "foot_right": UIGroup("foot_right", "🦶 Right Foot", "leg_right", ["Foot_R", "Toe_R"], order=51),
}

for idx, finger in enumerate(FINGER_NAMES):
    UI_GROUPS[f"finger_{finger.lower()}_left"] = UIGroup(
        f"finger_{finger.lower()}_left", f"  ├─ {FINGER_DISPLAY_NAMES[finger]}",
        "hand_left", [f"{finger}_{p}_L" for p in range(1, PHALANX_COUNT + 1)], order=22 + idx
    )
    UI_GROUPS[f"finger_{finger.lower()}_right"] = UIGroup(
        f"finger_{finger.lower()}_right", f"  ├─ {FINGER_DISPLAY_NAMES[finger]}",
        "hand_right", [f"{finger}_{p}_R" for p in range(1, PHALANX_COUNT + 1)], order=32 + idx
    )

for idx, toe in enumerate(TOE_NAMES):
    toe_key = toe.lower().replace("toe", "")
    UI_GROUPS[f"toe_{toe_key}_left"] = UIGroup(
        f"toe_{toe_key}_left", f"  ├─ {TOE_DISPLAY_NAMES[toe]}",
        "foot_left", [f"{toe}_{p}_L" for p in range(1, TOE_PHALANX_COUNT + 1)], "disable", 42 + idx
    )
    UI_GROUPS[f"toe_{toe_key}_right"] = UIGroup(
        f"toe_{toe_key}_right", f"  ├─ {TOE_DISPLAY_NAMES[toe]}",
        "foot_right", [f"{toe}_{p}_R" for p in range(1, TOE_PHALANX_COUNT + 1)], "disable", 52 + idx
    )

# =============================================================================
# ПРОПОРЦИИ
# =============================================================================

HEAD_BODY_PROPORTIONS: Dict[str, BodyPartProportions] = {
    "Head": BodyPartProportions("Head", (1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.7, 1.3), 100),
    "Neck": BodyPartProportions("Neck", (0.25, 0.45), (0.0, 0.9), (1.2, 1.2), (0.3, 0.7), 95),
    
    # =========================================================================
    # ПЛЕЧЕВОЙ ПОЯС
    # =========================================================================
    "Clavicle_L": BodyPartProportions(
        bone_name="Clavicle_L",
        size_relative_to_head=(0.4, 0.7),
        offset_from_head=(-0.8, 1.1),  # Левее и ниже шеи
        search_area_size=(1.5, 1.0),
        aspect_ratio=(0.2, 0.5),  # Горизонтальная
        priority=90
    ),
    
    "Clavicle_R": BodyPartProportions(
        bone_name="Clavicle_R",
        size_relative_to_head=(0.4, 0.7),
        offset_from_head=(0.8, 1.1),  # Правее и ниже шеи
        search_area_size=(1.5, 1.0),
        aspect_ratio=(0.2, 0.5),
        priority=90
    ),
    
    # =========================================================================
    # РУКИ
    # =========================================================================
    "Shoulder_L": BodyPartProportions(
        bone_name="Shoulder_L",
        size_relative_to_head=(0.6, 1.2),  # Плечо довольно крупное
        offset_from_head=(-1.3, 1.5),  # От шеи влево-вниз
        search_area_size=(2.0, 2.0),
        aspect_ratio=(0.25, 0.6),  # Вытянутое
        priority=85
    ),
    
    "Shoulder_R": BodyPartProportions(
        bone_name="Shoulder_R",
        size_relative_to_head=(0.6, 1.2),
        offset_from_head=(1.3, 1.5),  # От шеи вправо-вниз
        search_area_size=(2.0, 2.0),
        aspect_ratio=(0.25, 0.6),
        priority=85
    ),
    
    "Forearm_L": BodyPartProportions(
        bone_name="Forearm_L",
        size_relative_to_head=(0.5, 1.1),  # Чуть меньше плеча
        offset_from_head=(-1.5, 2.8),  # От плеча дальше
        search_area_size=(2.5, 2.5),
        aspect_ratio=(0.2, 0.5),
        priority=80
    ),
    
    "Forearm_R": BodyPartProportions(
        bone_name="Forearm_R",
        size_relative_to_head=(0.5, 1.1),
        offset_from_head=(1.5, 2.8),
        search_area_size=(2.5, 2.5),
        aspect_ratio=(0.2, 0.5),
        priority=80
    ),
    
    "Hand_L": BodyPartProportions(
        bone_name="Hand_L",
        size_relative_to_head=(0.3, 0.7),  # Кисть меньше
        offset_from_head=(-1.8, 4.0),  # От предплечья
        search_area_size=(2.5, 2.5),
        aspect_ratio=(0.4, 1.2),  # Может быть разной формы
        priority=75
    ),
    
    "Hand_R": BodyPartProportions(
        bone_name="Hand_R",
        size_relative_to_head=(0.3, 0.7),
        offset_from_head=(1.8, 4.0),
        search_area_size=(2.5, 2.5),
        aspect_ratio=(0.4, 1.2),
        priority=75
    ),
    
    # =========================================================================
    # ТОРС
    # =========================================================================
    "Spine_3": BodyPartProportions(
        bone_name="Spine_3",
        size_relative_to_head=(0.8, 1.4),  # Верхняя часть торса
        offset_from_head=(0.0, 1.8),  # Под шеей
        search_area_size=(2.0, 1.5),
        aspect_ratio=(0.5, 1.0),
        priority=88
    ),
    
    "Spine_2": BodyPartProportions(
        bone_name="Spine_2",
        size_relative_to_head=(0.8, 1.4),
        offset_from_head=(0.0, 2.5),
        search_area_size=(2.0, 1.5),
        aspect_ratio=(0.5, 1.0),
        priority=87
    ),
    
    "Spine_1": BodyPartProportions(
        bone_name="Spine_1",
        size_relative_to_head=(0.8, 1.4),
        offset_from_head=(0.0, 3.2),
        search_area_size=(2.0, 1.5),
        aspect_ratio=(0.5, 1.0),
        priority=86
    ),
    
    "Pelvis": BodyPartProportions(
        bone_name="Pelvis",
        size_relative_to_head=(0.9, 1.5),  # Таз широкий
        offset_from_head=(0.0, 4.0),  # Внизу торса
        search_area_size=(2.5, 1.5),
        aspect_ratio=(0.6, 1.2),
        priority=84
    ),
    
    "Root": BodyPartProportions(
        bone_name="Root",
        size_relative_to_head=(0.5, 1.0),
        offset_from_head=(0.0, 4.1),
        search_area_size=(2.0, 1.5),
        aspect_ratio=(0.5, 1.0),
        priority=83
    ),
    
    # =========================================================================
    # НОГИ
    # =========================================================================
    "Thigh_L": BodyPartProportions(
        bone_name="Thigh_L",
        size_relative_to_head=(0.7, 1.4),  # Бедро крупное
        offset_from_head=(-0.6, 4.5),  # От таза вниз-влево
        search_area_size=(2.5, 2.5),
        aspect_ratio=(0.25, 0.6),  # Вытянутое
        priority=82
    ),
    
    "Thigh_R": BodyPartProportions(
        bone_name="Thigh_R",
        size_relative_to_head=(0.7, 1.4),
        offset_from_head=(0.6, 4.5),  # От таза вниз-вправо
        search_area_size=(2.5, 2.5),
        aspect_ratio=(0.25, 0.6),
        priority=82
    ),
    
    "Calf_L": BodyPartProportions(
        bone_name="Calf_L",
        size_relative_to_head=(0.6, 1.3),  # Голень чуть меньше бедра
        offset_from_head=(-0.7, 6.0),  # От бедра вниз
        search_area_size=(2.5, 2.5),
        aspect_ratio=(0.2, 0.5),
        priority=78
    ),
    
    "Calf_R": BodyPartProportions(
        bone_name="Calf_R",
        size_relative_to_head=(0.6, 1.3),
        offset_from_head=(0.7, 6.0),
        search_area_size=(2.5, 2.5),
        aspect_ratio=(0.2, 0.5),
        priority=78
    ),
    
    "Foot_L": BodyPartProportions(
        bone_name="Foot_L",
        size_relative_to_head=(0.4, 0.9),  # Стопа меньше
        offset_from_head=(-0.8, 7.5),  # Внизу
        search_area_size=(2.0, 2.0),
        aspect_ratio=(0.3, 0.8),  # Зависит от угла
        priority=74
    ),
    
    "Foot_R": BodyPartProportions(
        bone_name="Foot_R",
        size_relative_to_head=(0.4, 0.9),
        offset_from_head=(0.8, 7.5),
        search_area_size=(2.0, 2.0),
        aspect_ratio=(0.3, 0.8),
        priority=74
    ),
    
    "Toe_L": BodyPartProportions(
        bone_name="Toe_L",
        size_relative_to_head=(0.2, 0.5),
        offset_from_head=(-0.9, 8.0),
        search_area_size=(1.5, 1.5),
        aspect_ratio=(0.5, 1.5),
        priority=70
    ),
    
    "Toe_R": BodyPartProportions(
        bone_name="Toe_R",
        size_relative_to_head=(0.2, 0.5),
        offset_from_head=(0.9, 8.0),
        search_area_size=(1.5, 1.5),
        aspect_ratio=(0.5, 1.5),
        priority=70
    ),
}

# =============================================================================
# MAPPING
# =============================================================================

COCO_TO_SKELETON: Dict[int, str] = {
    0: "Head", 5: "Shoulder_L", 6: "Shoulder_R",
    7: "Forearm_L", 8: "Forearm_R", 9: "Hand_L", 10: "Hand_R",
    11: "Thigh_L", 12: "Thigh_R", 13: "Calf_L", 14: "Calf_R",
    15: "Foot_L", 16: "Foot_R",
}

HAND_KEYPOINT_MAP: Dict[int, Tuple[str, int]] = {
    0: ("Hand", 0),
    1: ("Thumb", 1), 2: ("Thumb", 1), 3: ("Thumb", 2), 4: ("Thumb", 3),
    5: ("Index", 1), 6: ("Index", 1), 7: ("Index", 2), 8: ("Index", 3),
    9: ("Middle", 1), 10: ("Middle", 1), 11: ("Middle", 2), 12: ("Middle", 3),
    13: ("Ring", 1), 14: ("Ring", 1), 15: ("Ring", 2), 16: ("Ring", 3),
    17: ("Pinky", 1), 18: ("Pinky", 1), 19: ("Pinky", 2), 20: ("Pinky", 3),
}

# =============================================================================
# ЦВЕТА
# =============================================================================

BONE_COLORS: Dict[BoneCategory, Tuple[int, int, int]] = {
    BoneCategory.CORE: (255, 255, 0),
    BoneCategory.HEAD: (255, 0, 255),
    BoneCategory.ARM_L: (0, 255, 0),
    BoneCategory.ARM_R: (0, 180, 0),
    BoneCategory.HAND_L: (0, 255, 255),
    BoneCategory.HAND_R: (0, 180, 180),
    BoneCategory.LEG_L: (255, 100, 100),
    BoneCategory.LEG_R: (180, 70, 70),
    BoneCategory.FOOT_L: (255, 165, 0),
    BoneCategory.FOOT_R: (200, 130, 0),
}

OPENPOSE_FALLBACK = [("Neck", "Thigh_L"), ("Neck", "Thigh_R")]

# =============================================================================
# UI ГРУППЫ
# =============================================================================

@dataclass
class UIGroup:
    key: str
    display_name: str
    parent_group: Optional[str] = None
    bones: List[str] = field(default_factory=list)
    default: str = "enable"
    order: int = 0


UI_GROUPS: Dict[str, UIGroup] = {
    "body": UIGroup("body", "🦴 Body", bones=["Root", "Pelvis", "Spine_1", "Spine_2", "Spine_3"], order=0),
    "pelvis_spine": UIGroup("pelvis_spine", "├─ Pelvis & Spine", "body", ["Pelvis", "Spine_1", "Spine_2", "Spine_3"], order=1),
    "head": UIGroup("head", "👤 Head", bones=["Neck", "Head"], order=10),
    "arm_left": UIGroup("arm_left", "💪 Left Arm", bones=["Clavicle_L", "Shoulder_L", "Forearm_L"], order=20),
    "hand_left": UIGroup("hand_left", "✋ Left Hand", "arm_left", ["Hand_L"], order=21),
    "arm_right": UIGroup("arm_right", "💪 Right Arm", bones=["Clavicle_R", "Shoulder_R", "Forearm_R"], order=30),
    "hand_right": UIGroup("hand_right", "✋ Right Hand", "arm_right", ["Hand_R"], order=31),
    "leg_left": UIGroup("leg_left", "🦵 Left Leg", bones=["Thigh_L", "Calf_L"], order=40),
    "foot_left": UIGroup("foot_left", "🦶 Left Foot", "leg_left", ["Foot_L", "Toe_L"], order=41),
    "leg_right": UIGroup("leg_right", "🦵 Right Leg", bones=["Thigh_R", "Calf_R"], order=50),
    "foot_right": UIGroup("foot_right", "🦶 Right Foot", "leg_right", ["Foot_R", "Toe_R"], order=51),
}

for idx, finger in enumerate(FINGER_NAMES):
    UI_GROUPS[f"finger_{finger.lower()}_left"] = UIGroup(
        f"finger_{finger.lower()}_left", f"  ├─ {FINGER_DISPLAY_NAMES[finger]}",
        "hand_left", [f"{finger}_{p}_L" for p in range(1, PHALANX_COUNT + 1)], order=22 + idx
    )
    UI_GROUPS[f"finger_{finger.lower()}_right"] = UIGroup(
        f"finger_{finger.lower()}_right", f"  ├─ {FINGER_DISPLAY_NAMES[finger]}",
        "hand_right", [f"{finger}_{p}_R" for p in range(1, PHALANX_COUNT + 1)], order=32 + idx
    )

for idx, toe in enumerate(TOE_NAMES):
    toe_key = toe.lower().replace("toe", "")
    UI_GROUPS[f"toe_{toe_key}_left"] = UIGroup(
        f"toe_{toe_key}_left", f"  ├─ {TOE_DISPLAY_NAMES[toe]}",
        "foot_left", [f"{toe}_{p}_L" for p in range(1, TOE_PHALANX_COUNT + 1)], "disable", 42 + idx
    )
    UI_GROUPS[f"toe_{toe_key}_right"] = UIGroup(
        f"toe_{toe_key}_right", f"  ├─ {TOE_DISPLAY_NAMES[toe]}",
        "foot_right", [f"{toe}_{p}_R" for p in range(1, TOE_PHALANX_COUNT + 1)], "disable", 52 + idx
    )

# =============================================================================
# COCO KEYPOINTS
# =============================================================================

COCO_TO_SKELETON: Dict[int, str] = {
    0: "Head",
    5: "Shoulder_L",
    6: "Shoulder_R",
    7: "Forearm_L",
    8: "Forearm_R",
    9: "Hand_L",
    10: "Hand_R",
    11: "Thigh_L",
    12: "Thigh_R",
    13: "Calf_L",
    14: "Calf_R",
    15: "Foot_L",
    16: "Foot_R",
}

HAND_KEYPOINT_MAP: Dict[int, Tuple[str, int]] = {
    0: ("Hand", 0),
    1: ("Thumb", 1), 2: ("Thumb", 1), 3: ("Thumb", 2), 4: ("Thumb", 3),
    5: ("Index", 1), 6: ("Index", 1), 7: ("Index", 2), 8: ("Index", 3),
    9: ("Middle", 1), 10: ("Middle", 1), 11: ("Middle", 2), 12: ("Middle", 3),
    13: ("Ring", 1), 14: ("Ring", 1), 15: ("Ring", 2), 16: ("Ring", 3),
    17: ("Pinky", 1), 18: ("Pinky", 1), 19: ("Pinky", 2), 20: ("Pinky", 3),
}

# =============================================================================
# ЦВЕТА
# =============================================================================

BONE_COLORS: Dict[BoneCategory, Tuple[int, int, int]] = {
    BoneCategory.CORE: (255, 255, 0),
    BoneCategory.HEAD: (255, 0, 255),
    BoneCategory.ARM_L: (0, 255, 0),
    BoneCategory.ARM_R: (0, 180, 0),
    BoneCategory.HAND_L: (0, 255, 255),
    BoneCategory.HAND_R: (0, 180, 180),
    BoneCategory.LEG_L: (255, 100, 100),
    BoneCategory.LEG_R: (180, 70, 70),
    BoneCategory.FOOT_L: (255, 165, 0),
    BoneCategory.FOOT_R: (200, 130, 0),
}

OPENPOSE_FALLBACK = [("Neck", "Thigh_L"), ("Neck", "Thigh_R")]

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

@dataclass
class RefinementConfig:
    """Настройки уточнения скелета"""
    use_background_removal: bool = True
    use_depth: bool = True
    use_edge: bool = True
    use_contour_analysis: bool = True
    depth_weight: float = 0.3
    edge_weight: float = 0.4
    contour_weight: float = 0.5
    edge_threshold_low: int = 30
    edge_threshold_high: int = 100
    depth_model: str = "midas_v21_small"
    edge_model: str = "canny"
    contour_search_radius: int = 25
    min_contour_confidence: float = 0.3
    min_point_confidence: float = 0.1
    min_contour_area: int = 100
    ellipse_fit_quality: float = 0.6
    convexity_threshold: float = 0.4  # Порог для выпуклости (0.4 = умеренная выпуклость)
    depth_tolerance: float = 0.2  # Допуск по глубине относительно тела
    
# =============================================================================
# ЦЕПОЧКИ КОНЕЧНОСТЕЙ
# =============================================================================

# Последовательности костей в цепочках (от торса к конечностям)
LIMB_CHAINS = {
    "arm_L": ["Clavicle_L", "Shoulder_L", "Forearm_L", "Hand_L"],
    "arm_R": ["Clavicle_R", "Shoulder_R", "Forearm_R", "Hand_R"],
    "leg_L": ["Thigh_L", "Calf_L", "Foot_L"],
    "leg_R": ["Thigh_R", "Calf_R", "Foot_R"],

}

"""
Базовый класс для работы с 3D данными (нормали, глубина, контуры)
Содержит проверенные методы получения и вывода данных
"""

class DWPose3DData:
    """Базовый класс для работы с 3D данными"""
    
    @staticmethod
    def get_depth_data(depth_map):
        """Получение данных глубины"""
        if isinstance(depth_map, torch.Tensor):
            depth_np = depth_map.cpu().numpy()
        else:
            depth_np = depth_map
            
        # Нормализация глубины
        if depth_np.max() > 0:
            depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        else:
            depth_normalized = depth_np
            
        return depth_normalized
    
    @staticmethod
    def get_normal_data(normal_map):
        """Получение данных нормалей"""
        if isinstance(normal_map, torch.Tensor):
            normal_np = normal_map.cpu().numpy()
        else:
            normal_np = normal_map
            
        # Нормализация нормалей [-1, 1] -> [0, 1]
        normal_normalized = (normal_np + 1.0) / 2.0
        return np.clip(normal_normalized, 0, 1)
    
    @staticmethod
    def get_edge_data(edge_map):
        """Получение данных контуров"""
        if isinstance(edge_map, torch.Tensor):
            edge_np = edge_map.cpu().numpy()
        else:
            edge_np = edge_map
            
        # Бинаризация контуров
        if edge_np.max() > 1:
            edge_np = edge_np / 255.0
            
        return edge_np
    
    @staticmethod
    def prepare_output(data, channels=3):
        """
        Подготовка данных для вывода
        Args:
            data: numpy array данных
            channels: количество каналов (1 или 3)
        Returns:
            torch.Tensor в формате ComfyUI [B, H, W, C]
        """
        # Убедимся что данные в правильном диапазоне
        data = np.clip(data, 0, 1)
        
        # Приведение к нужному количеству каналов
        if data.ndim == 2:  # [H, W]
            if channels == 3:
                data = np.stack([data] * 3, axis=-1)  # [H, W, 3]
            else:
                data = data[..., None]  # [H, W, 1]
        elif data.ndim == 3:
            if data.shape[-1] == 1 and channels == 3:
                data = np.repeat(data, 3, axis=-1)
            elif data.shape[-1] == 3 and channels == 1:
                data = data.mean(axis=-1, keepdims=True)
        
        # Добавляем batch dimension если нужно
        if data.ndim == 3:
            data = data[None, ...]  # [1, H, W, C]
        
        # Конвертируем в torch tensor
        output = torch.from_numpy(data.astype(np.float32))
        
        return output
