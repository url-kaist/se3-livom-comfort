// Online backend: global plane BA (math from hku-mars/HBA a0cdd47) re-solved periodically while scans arrive, never
// at the end. Keyframes (25 scans merged) accumulate; every `every` new keyframes the worker solves all of them from
// the LIO poses (cold start, as the batch does), and the all-pair relative poses enter an incremental PGO (iSAM2)
// together with per-scan odometry factors and stationary-node gravity factors (accelerometer during 4 s still chunks,
// bias and world up from the LIO state). With loop_radius > 0 every new top keyframe is paired with the nearest earlier
// one it revisits (>= loop_gap s back, each target once), point-to-plane ICP of the two keyframe neighbourhoods is gated
// on residual, overlap and motion, and the accepted loop enters the PGO as a Huber factor; the BA of a round then starts
// from the LIO chain closed by those loops instead of the raw LIO poses (the offline loop PGO + HBA order). finish() runs
// no BA: the pose graph estimate at that moment is the output, and a round still running is discarded.
#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace se3_lio {
namespace backend {

struct Params {
    double voxel_size = 1.0;
    double downsample_size = 0.2;  // top-layer keyframe cloud, applied once
    double eigen_ratio = 0.01;     // plane threshold of the top layer is 2x this (hba.cpp global_ba)
    double reject_ratio = 0.05;
    int max_iter = 10;             // stops earlier when the residual changes < 5 %
    int layers = 3;                // keyframe stride = 5^(layers-1) scans
    int threads = 8;               // one BA round (cut / recut / solve)
    int every = 4;                 // new top keyframes per round
    std::vector<double> hess_const = {1969, 1713, 2248, 4.17, 6.67, 10.1};  // scan-pair PGO weight (window BA Hessian median)
    double gravity_sigma_deg = 0.02;  // stationary-chunk attitude factor: sigma of one 4 s chunk (per node x sqrt(n)); 0 = off
    double loop_radius = 0;           // m: loop candidates between top keyframes that revisit within this distance (0 = off)
    double loop_gap = 30;             // s: the two keyframes of a loop are at least this far apart in time
    double loop_huber = 2.5;          // Huber threshold of the loop factors (0 = plain Gaussian)
    std::string dump_dir;          // if set, every round's BA poses go to <dump_dir>/round_<kfs>.txt (regression checks)
};

struct Pose {
    Eigen::Matrix3d R;
    Eigen::Vector3d p;
};

// Point-to-plane ICP of `src` onto `tgt` (float32 xyz rows, both in their own frames) from T0 = pose of the source frame
// in the target frame: refined T, information in Pose3 tangent order (rotation, translation) and the gate statistics.
struct IcpResult {
    Eigen::Matrix4d T;
    Eigen::Matrix<double, 6, 6> info;
    double rms, matched, inlier;
    int n, iters;
};
IcpResult loop_icp(const float *src, int ns, const float *tgt, int nt, const Eigen::Matrix4d &T0, int iters = 40, double coarse = 1.0);

struct RoundStat {
    int round;
    int start_scan;  // scans pushed when the round started
    int kfs;         // top keyframes solved
    int iters;
    double ba_ms;
    double pgo_ms;   // replacing the top-layer factors in iSAM2
    int done_scan;   // scans pushed when the result entered the PGO (-1: discarded at finish)
    double rss_mb;   // resident memory of the process right after the BA
};

class OnlineBackend {
public:
    explicit OnlineBackend(const Params &params);
    ~OnlineBackend();
    // Deskewed body-frame scan (float32 xyz, n rows) with its LIO pose (quaternion xyzw), stamp, the mean accelerometer
    // reading of its IMU rows (body frame, NaN = none) and the LIO's accelerometer bias and world gravity (down).
    void push(const Eigen::Vector4d &q_xyzw, const Eigen::Vector3d &p, const float *xyz, int n, double stamp,
              const Eigen::Vector3d &acc, const Eigen::Vector3d &ba, const Eigen::Vector3d &grav);
    // The pose graph estimate now: one pose per pushed scan (scans past the last odometry window follow the LIO
    // relative motion). No BA runs here; a round in flight is waited for but discarded (see pending_ms).
    std::vector<Pose> finish();
    const std::vector<RoundStat> &stats() const;
    double pending_ms() const;  // time finish() spent waiting for the discarded round
    int loops() const;          // accepted loop factors so far

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace backend
}  // namespace se3_lio
