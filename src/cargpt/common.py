from enum import StrEnum, auto, unique


@unique
class CameraName(StrEnum):
    cam_front_center = auto()
    cam_front_left = auto()
    cam_front_right = auto()
    cam_left_forward = auto()
    cam_right_forward = auto()
    cam_left_backward = auto()
    cam_right_backward = auto()
    cam_rear = auto()
