import torch
import torch.nn.functional as F
from kornia.geometry.camera import project_points
from kornia.geometry.conversions import (
    Rt_to_matrix4x4,
    axis_angle_to_rotation_matrix,
    normalize_pixel_coordinates,
)
from kornia.geometry.depth import depth_to_3d
from kornia.geometry.linalg import transform_points


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert all(condition), "wrong size for {}, expected {}, got  {}".format(
        input_name, "x".join(expected), list(input.size())
    )


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack(
        [cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1
    ).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack(
        [cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1
    ).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack(
        [ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1
    ).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode="euler"):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == "euler":
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == "quat":
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def pose_to_mat(vec, rotation_mode="euler"):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 4, 4]
    """
    t = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    R = axis_angle_to_rotation_matrix(rot)  # [B, 3, 3]
    transform_mat = Rt_to_matrix4x4(R, t)
    return transform_mat


def inverse_warp(
    ref_img, tgt_disparity, ref_disparity, pose, camera_model, padding_mode="zeros"
):
    """
    Inverse warp a source image to the target image plane.
    Args:
        ref_img: the reference image (where to sample pixels) -- [B, 3, H, W]
        tgt_depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the refence depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to refence -- [B, 6]
    Returns:
        projected_img: reference image warped to the target image plane
        projected_depth: sampled reference depth from reference image
        computed_depth: computed depth of reference image using the target depth and pose
    """
    check_sizes(ref_img, "img", "B3HW")
    check_sizes(tgt_disparity, "disparity", "B1HW")
    check_sizes(ref_disparity, "ref_disparity", "B1HW")
    check_sizes(pose, "pose", "B6")

    B, _, H, W = ref_img.size()
    # TODO: can be cached for known batch size
    grid = torch.stack(
        torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy"), dim=-1
    ).to(tgt_disparity.device)

    tgt_depth = 1 / tgt_disparity
    camera_model = camera_model.to(tgt_depth)
    points_3d_tgt = camera_model.unproject(grid, tgt_depth.permute(0, 2, 3, 1))

    # transform points from source to destination
    # apply transformation to the 3d points
    camera_transform = pose_to_mat(pose)  # [B,3,4]
    points_3d_ref = transform_points(
        camera_transform[:, None], points_3d_tgt
    )  # BxHxWx3
    points_2d_ref = camera_model.project(points_3d_ref)

    # normalize points between [-1 / 1]
    points_2d_ref_norm = normalize_pixel_coordinates(points_2d_ref, H, W)  # BxHxWx2

    # X=0, Y=1, Z=2
    computed_depth = points_3d_ref[:, :, :, [2]].permute(0, 3, 1, 2)  # Bx1xHxW
    computed_disparity = 1 / (computed_depth + 1e-7)

    projected_img = F.grid_sample(ref_img, points_2d_ref_norm, align_corners=False)

    projected_disparity = F.grid_sample(
        ref_disparity,
        points_2d_ref_norm,
        padding_mode=padding_mode,
        align_corners=False,
    )

    # NOTE: This projected and computed disparities are for pixels in tgt_img

    return projected_img, projected_disparity, computed_disparity


# https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/depth.html#warp_frame_depth
def inverse_warp_kornia(
    ref_img, tgt_disparity, ref_disparity, pose, intrinsics, padding_mode="zeros"
):
    """
    Inverse warp a source image to the target image plane.
    Args:
        ref_img: the reference image (where to sample pixels) -- [B, 3, H, W]
        tgt_depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the refence depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to refence -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: reference image warped to the target image plane
        projected_depth: sampled reference depth from reference image
        computed_depth: computed depth of reference image using the target depth and pose
    """
    check_sizes(ref_img, "img", "B3HW")
    check_sizes(tgt_disparity, "disparity", "B1HW")
    check_sizes(ref_disparity, "ref_disparity", "B1HW")
    check_sizes(pose, "pose", "B6")
    check_sizes(intrinsics, "intrinsics", "B33")

    B, _, H, W = ref_img.size()
    tgt_depth = 1 / tgt_disparity

    camera_transform = pose_to_mat(pose)  # [B,3,4]

    points_3d_tgt = depth_to_3d(tgt_depth, intrinsics)  # Bx3xHxW

    # transform points from source to destination
    points_3d_tgt = points_3d_tgt.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_ref = transform_points(
        camera_transform[:, None], points_3d_tgt
    )  # BxHxWx3

    # project back to pixels
    camera_matrix_tmp = intrinsics[:, None, None]  # Bx1x1xHxW
    points_2d_ref = project_points(points_3d_ref, camera_matrix_tmp)  # BxHxWx2

    # normalize points between [-1 / 1]
    points_2d_ref_norm = normalize_pixel_coordinates(points_2d_ref, H, W)  # BxHxWx2

    # X=0, Y=1, Z=2
    computed_depth = points_3d_ref[:, :, :, [2]].permute(0, 3, 1, 2)  # Bx1xHxW
    computed_disparity = 1 / (computed_depth + 1e-7)

    projected_img = F.grid_sample(ref_img, points_2d_ref_norm, align_corners=False)

    projected_disparity = F.grid_sample(
        ref_disparity,
        points_2d_ref_norm,
        padding_mode=padding_mode,
        align_corners=False,
    )

    # NOTE: This projected and computed disparities are for pixels in tgt_img

    return projected_img, projected_disparity, computed_disparity
