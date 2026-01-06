import numpy as np

class KalmanFilter:
    def __init__(self, n_dim_state, n_dim_obs, initial_state, initial_covariance, dt):
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs

        # Initialize state estimate and covariance
        self.state = initial_state
        self.covariance = initial_covariance

        # Initialize transition matrix (constant acceleration model)
        self.transition_matrix = np.array([[1, 0, dt, 0, 0.5*(dt**2), 0],
                                           [0, 1, 0, dt, 0, 0.5*(dt**2)],
                                           [0, 0, 1, 0, dt, 0],
                                           [0, 0, 0, 1, 0, dt],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]])

        # Initialize observation matrix
        self.observation_matrix = np.array([[1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0]])

        # Initialize process noise covariance matrix
        # self.process_noise_cov = np.eye(n_dim_state) * 1e-5
        self.process_noise_cov = np.array([[0.25*(dt**4),0,0.5*(dt**3),0,0.5*(dt**2),0],
                                           [0, 0.25*(dt**4), 0, 0.5*(dt**3), 0, 0.5*(dt**2)],
                                           [0.5*(dt**3), 0, dt**2,0,dt,0],
                                           [0, 0.5*(dt**3), 0, dt**2, 0, dt],
                                          [0.5*(dt**2), 0, dt, 0, 1, 0],
                                           [0,0.5*(dt**2),0,dt,0,1]])

        # Initialize observation noise covariance matrix
        self.observation_noise_cov = np.eye(n_dim_obs) * 4

    def predict(self):
        # Predict next state and covariance
        self.state = np.dot(self.transition_matrix, self.state)
        self.covariance = np.dot(np.dot(self.transition_matrix, self.covariance),
                                 self.transition_matrix.T) + self.process_noise_cov

    def update(self, observation):
        # Compute Kalman gain
        kalman_gain = np.dot(np.dot(self.covariance, self.observation_matrix.T),
                             np.linalg.inv(np.dot(np.dot(self.observation_matrix, self.covariance),
                                                  self.observation_matrix.T) + self.observation_noise_cov))

        # Update state estimate and covariance
        self.state = self.state + np.dot(kalman_gain, (observation - np.dot(self.observation_matrix, self.state)))
        self.covariance = self.covariance - np.dot(np.dot(kalman_gain, self.observation_matrix), self.covariance)

        return self.state

