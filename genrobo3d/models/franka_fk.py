import numpy as np
from scipy.spatial.transform import Rotation as R

# Denavit-Hartenberg parameters for Franka Panda (modified DH convention)
# These are based on public documentation and typical usage
d = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]       # link offsets
a = [0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0]       # link lengths
alpha = [-np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0.0]  # link twists

def homogeneous_transformation_matrix(alpha, a, d, theta):
    """
    Returns the homogeneous transformation matrix using modified DH parameters.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    A = np.array([
        [ct, -st, 0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0, 0, 0, 1]
    ])

    return A

def forward_kinematics(joint_angles):
    """
    Computes the forward kinematics for the Franka Panda arm.

    Args:
        joint_angles: list of 7 joint angles in radians.

    Returns:
        A 4x4 numpy array representing the end-effector pose.
    """
    if len(joint_angles) != 7:
        raise ValueError("Expected 7 joint angles for Franka Panda.")

    T = np.eye(4)

    for i in range(7):
        A_i = homogeneous_transformation_matrix(alpha[i], a[i], d[i], joint_angles[i])
        T = T @ A_i

    return T

def pose_to_position_quaternion(T):
    """
    Extracts position and orientation (quaternion) from a transformation matrix.

    Args:
        T: 4x4 transformation matrix.

    Returns:
        position: [x, y, z]
        quaternion: [qx, qy, qz, qw]
    """
    position = T[:3, 3]
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # returns [x, y, z, w]
    return position, quat

def get_pose_from_joint_angles(joint_angles):
    """
    Full pipeline: joint angles to position and quaternion.
    """
    T = forward_kinematics(joint_angles)
    pos, quat = pose_to_position_quaternion(T)
    return pos, quat

# Example joint configuration
joint_angles = [0.0, -np.pi/4, 0.0, -np.pi/2, 0.0, np.pi/3, 0.0]

# Compute pose
position, quaternion = get_pose_from_joint_angles(joint_angles)

# Output
print("End-Effector Position:", position)
print("End-Effector Orientation (quaternion):", quaternion)  # [x, y, z, w]
