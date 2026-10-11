"""Loop ICP of the online backend on synthetic data (three walls)."""

import numpy as np
from scipy.spatial.transform import Rotation

from se3_lio import se3_lio_pybind as lio


def pose(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def three_walls(seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-5, 5, (3, 6000, 2))
    walls = [
        np.c_[u[0], np.zeros(6000)],
        np.c_[u[1, :, 0], np.full(6000, 4.0), u[1, :, 1]],
        np.c_[np.full(6000, -3.0), u[2]],
    ]
    return np.concatenate(walls), rng


def test_loop_icp_recovers_a_small_offset_between_three_walls():
    clean, rng = three_walls()
    target = clean + rng.normal(0, 0.005, clean.shape)
    T_true = pose([0.0, 0.0, np.radians(2)], [0.05, -0.03, 0.02])
    source = (clean - T_true[:3, 3]) @ T_true[:3, :3]
    T, info, st = lio._loop_icp(source.astype(np.float32), target.astype(np.float32), np.eye(4))
    assert np.linalg.norm(T[:3, 3] - T_true[:3, 3]) < 0.003
    assert np.degrees(Rotation.from_matrix(T[:3, :3].T @ T_true[:3, :3]).magnitude()) < 0.1
    assert st["rms"] < 0.01 and st["inlier"] > 0.95 and st["matched"] > 0.9 and st["n"] > 10000
    assert np.all(np.linalg.eigvalsh(info) > 0)


def test_loop_icp_from_a_wrong_start_fails_the_gates():
    clean, rng = three_walls()
    target = clean + rng.normal(0, 0.005, clean.shape)
    T0 = pose([0.0, 0.0, 0.0], [3.0, 3.0, 0.0])  # far from any overlap at the 1 m correspondence bound
    T, info, st = lio._loop_icp(clean.astype(np.float32), target.astype(np.float32), T0)
    assert st["matched"] < 0.5 or st["rms"] > 0.06 or st["inlier"] < 0.7
