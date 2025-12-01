import numpy as np

def update(x, P, Z, H, R):
    '''
    Perform the Kalman Filter update step.

    Arguments:
    x: State estimate vector
    P: Estimate covariance matrix
    Z: Measurement vector
    H: Observation matrix
    R: Measurement noise covariance matrix

    Returns:
    x: Updated state estimate vector
    P: Updated estimate covariance matrix
    '''

    y = Z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye((K @ H).shape[0]) - K @ H) @ P

    return x, P
    
def predict(x, P, F, u):
    '''
    Perform the Kalman Filter prediction step.

    Arguments:
    x: State estimate vector
    P: Estimate covariance matrix
    F: State transition matrix
    u: Control input vector

    Returns:
    x: Predicted state estimate vector
    P: Predicted estimate covariance matrix
    '''

    x = F @ x + u
    P = F @ P @ F.T

    return x, P

def kalman_filter(x, P, Z, F, H, R, u):
    '''
    Perform Kalman filtering over a series of measurements.

    Arguments:
    x: Initial state estimate vector
    P: Initial estimate covariance matrix
    Z: Measurement vectors
    F: State transition matrix
    H: Observation matrix
    R: Measurement noise covariance matrix
    u: Control input vector

    Returns:
    x_current: Found current states
    x_predict: Found predicted states
    '''

    Z = np.array(Z).reshape(-1, 1)
    if Z.shape[0] != 0:
        x, P = update(x, P, Z, H, R)
    x, P = predict(x, P, F, u)
    
    return x, P
