import numpy as np
from typing import Tuple, List

KALMAN_STATE_SPACE_DIM = 8
KALMAN_MEASUREMENT_SPACE_DIM = 4

KFDataStateSpace = Tuple[np.ndarray, np.ndarray]  # mean, covariance
KFDataMeasurementSpace = Tuple[np.ndarray, np.ndarray]  # mean, covariance
DetVec = np.ndarray
KFStateSpaceVec = np.ndarray
KFStateSpaceMatrix = np.ndarray
KFMeasSpaceVec = np.ndarray
KFMeasSpaceMatrix = np.ndarray


class KalmanFilter:
    chi2inv95 = [0, 3.8415, 5.9915, 7.8147,
                 9.4877, 11.070, 12.592, 14.067,
                 15.507, 16.919]

    def __init__(self):
        dt = 1
        self._init_pos_weight = 5.0
        self._init_vel_weight = 15.0
        self._std_factor_acceleration = 50.25
        self._std_offset_acceleration = 100.5
        self._std_factor_detection = 0.10
        self._min_std_detection = 4.0
        self._std_factor_motion_compensated_detection = 0.14
        self._min_std_motion_compensated_detection = 5.0
        self._velocity_coupling_factor = 0.6
        self._velocity_half_life = 2

        self._init_kf_matrices(dt)

    def _init_kf_matrices(self, dt: float):
        self._measurement_matrix = np.eye(KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM)
        self._state_transition_matrix = np.eye(KALMAN_STATE_SPACE_DIM)

        for i in range(4):
            self._state_transition_matrix[i, i + 4] = self._velocity_coupling_factor * dt
            self._state_transition_matrix[i, (i + 2) % 4 + 4] = (1.0 - self._velocity_coupling_factor) * dt
            self._state_transition_matrix[i + 4, i + 4] = pow(0.5, dt / self._velocity_half_life)

        self._process_noise_covariance = np.eye(KALMAN_STATE_SPACE_DIM)
        for i in range(4):
            self._process_noise_covariance[i, i] = pow(dt, 4) / 4 + pow(dt, 2)
            self._process_noise_covariance[i, i + 4] = pow(dt, 3) / 2
            self._process_noise_covariance[i + 4, i] = pow(dt, 3) / 2
            self._process_noise_covariance[i + 4, i + 4] = pow(dt, 2)

    def initiate(self, det: DetVec) -> KFDataStateSpace:
        mean = np.zeros(KALMAN_STATE_SPACE_DIM)
        mean[:4] = det[:4]

        w, h = det[2], det[3]
        std_dev = np.concatenate([
            np.maximum(self._init_pos_weight * self._std_factor_detection * np.array([w, h, w, h]), self._min_std_detection * np.ones(4)),
            np.maximum(self._init_vel_weight * self._std_factor_detection * np.array([w, h, w, h]), self._min_std_detection * np.ones(4))
        ])

        covariance = np.diag(std_dev ** 2)
        return mean, covariance

    def predict(self, mean: KFStateSpaceVec, covariance: KFStateSpaceMatrix):
        std = self._std_factor_acceleration * max(mean[2], mean[3]) + self._std_offset_acceleration
        motion_cov = (std ** 2) * self._process_noise_covariance

        mean = self._state_transition_matrix @ mean
        covariance = (self._state_transition_matrix @ covariance @
                      self._state_transition_matrix.T + motion_cov)
        return mean, covariance

    def project(self, mean: KFStateSpaceVec, covariance: KFStateSpaceMatrix,
                motion_compensated: bool = False) -> KFDataMeasurementSpace:
        std_factor = self._std_factor_motion_compensated_detection if motion_compensated else self._std_factor_detection
        min_std = self._min_std_motion_compensated_detection if motion_compensated else self._min_std_detection

        std = np.array([
            max(std_factor * mean[2], min_std),
            max(std_factor * mean[3], min_std),
            max(std_factor * mean[2], min_std),
            max(std_factor * mean[3], min_std)
        ])

        measurement_cov = np.diag(std ** 2)
        mean_projected = self._measurement_matrix @ mean
        covariance_projected = (self._measurement_matrix @ covariance @ self._measurement_matrix.T + measurement_cov)

        return mean_projected, covariance_projected

    def update(self, mean: KFStateSpaceVec, covariance: KFStateSpaceMatrix,
               measurement: DetVec) -> KFDataStateSpace:
        projected_mean, projected_covariance = self.project(mean, covariance)

        B = (covariance @ self._measurement_matrix.T).T
        kalman_gain = np.linalg.solve(projected_covariance, B).T

        innovation = measurement - projected_mean
        mean_updated = mean + kalman_gain @ innovation
        covariance_updated = covariance - kalman_gain @ projected_covariance @ kalman_gain.T

        return mean_updated, covariance_updated


    def multi_predict(self, means: List[KFStateSpaceVec],
                  covariances: List[KFStateSpaceMatrix]) -> Tuple[List[KFStateSpaceVec], List[KFStateSpaceMatrix]]:
        updated_means = []
        updated_covariances = []
        for mean, covariance in zip(means, covariances):
            mean, covariance = self.predict(mean, covariance)
            updated_means.append(mean)
            updated_covariances.append(covariance)
        return updated_means, updated_covariances


    def gating_distance(self, mean: KFStateSpaceVec, covariance: KFStateSpaceMatrix,
                        measurements: List[DetVec], only_position: bool = False) -> np.ndarray:
        projected_mean, projected_covariance = self.project(mean, covariance)

        if only_position:
            projected_mean[2:] = 0
            projected_covariance[2:, 2:] = 0

        try:
            L = np.linalg.cholesky(projected_covariance)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(projected_covariance + np.eye(projected_covariance.shape[0]) * 1e-3)

        distances = np.zeros(len(measurements))
        for i, measurement in enumerate(measurements):
            diff = measurement - projected_mean
            y = np.linalg.solve(L, diff)
            distances[i] = np.sum(y ** 2)

        return distances
