import torch
import torch.nn.functional as F


'''Euler angle representation'''
def euler2mat(angle):
    """Convert euler angles and translation to transformation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (a, b, y) in radians -- size = [B, 6]
               [:,:3] is euler angle
               [:,3:] is translation
    Returns:
        Transformation matrix -- size = [B, 4, 4]
    """
    B = angle.size(0)
    device = angle.device

    x, y, z = angle[:, 1], angle[:, 0], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = zmat @ ymat @ xmat  #xmat @ ymat @ zmat  # [B,3,3]

    res = torch.eye(4).unsqueeze(0).repeat(B,1,1).to(device)
    res[:,:3,:3] = rotMat
    res[:,:3, 3] = angle[:,3:]
    return res


'''Rot 9D representation'''
def rot9d2mat(x):
    '''x in shape [B,12], [,:9] is the 9D representation of rotation, [,9:] is translation'''
    device = x.device
    B = x.shape[0]

    rotMat = symmetric_orthogonalization(x[:,:9])  # [B,3,3]
    res = torch.eye(4).unsqueeze(0).repeat(B,1,1).to(device)
    res[:,:3,:3] = rotMat
    res[:,:3, 3] = x[:,9:]
    return res


def symmetric_orthogonalization(x):
  """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.
  x: should have size [batch_size, 9]
  Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
  """
  m = x.view(-1, 3, 3)
  u, s, v = torch.svd(m)
  vt = torch.transpose(v, 1, 2)
  det = torch.det(torch.matmul(u, vt))
  det = det.view(-1, 1, 1)
  vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
  r = torch.matmul(u, vt)
  return r


  '''Rot 6D representation'''
def rot6d2mat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    """
    device = x.device
    B = x.shape[0]

    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    rotMat = torch.stack((b1, b2, b3), dim=-1)  # [B,3,3]

    res = torch.eye(4).unsqueeze(0).repeat(B,1,1).to(device)
    res[:,:3,:3] = rotMat
    res[:,:3, 3] = x[:,6:]
    return res


'''Quaternion representation'''
def quat2mat(x):
    B = x.shape[0]
    device = x.device

    rotMat = quat2mat_transform(x[:,:4])  # [B,3,3]
    
    res = torch.eye(4).unsqueeze(0).repeat(B,1,1).to(device)
    res[:,:3,:3] = rotMat
    res[:,:3, 3] = x[:,4:]
    return res


def quat2mat_transform(quat):
    """Convert quaternion coefficients to rotation matrix.
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def mat2quat(x):
    # x: SE3 matrix in shape [B,4,4]
    trans = x[:,:3,3]
    rot = x[:,:3,:3]
    quat = mat2quat_transform(rot)
    return torch.cat([quat, trans], dim=1)


def mat2quat_transform(rotation_matrix, eps=1e-6):
    """Convert 3x3 rotation matrix to 4d quaternion vector"""
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q
