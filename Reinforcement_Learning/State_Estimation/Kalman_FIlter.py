
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
np.random.seed(42)
def one_D():
    np.random.seed(32)
    del_t = 0.01 
    Vs = 3000  

    At = np.array([[1, del_t], [0, 1]]) 
    Bt = np.array([[0.5 * (del_t ** 2)], [del_t]])  
    Ct = np.array([[2 / Vs, 0]]) 

    X = np.array([[0], [0]]) 
    Q = np.diag([0.1**2, 0.5**2])
    P = np.array([[10**-4, 0], [0, 10**-4]]) 
    R = np.array([[0.01**2]])  

    def control_input(t):
        if t < 0.25:
            return np.array([[400]])
        elif 3.0 < t < 3.25:
            return np.array([[-400]])
        else:
            return np.array([[0]])
    t = 0
    time_series = [0]
    K_gains=[[0,0]]
    X_Predictions = [X.flatten()]
    P_Predictions = [X.flatten()]
    X_Corrections = [X.flatten()]
    P_Corrections = [X.flatten()]
    X_grounds=[X.flatten()]

    X_ground=X
    for i in range(325):
        t += del_t 
        time_series.append(t)
        Ut = control_input(t)

        # Prediction step
        motion_noise = np.random.multivariate_normal([0, 0], Q).reshape(2, 1)
        X_ground = np.matmul(At, X_ground) + np.matmul(Bt, Ut)+motion_noise
        measurement_noise = np.random.normal(0, np.sqrt(R), size=(1, 1)) 
        Zt = (2 * X_ground[0, 0]) / Vs + measurement_noise 

        X_pred=np.matmul(At,X)+np.matmul(Bt,Ut)
        P_pred=np.matmul(At,np.matmul(P,At.T))+Q

        # Kalman gain
        S=np.linalg.inv(np.matmul(Ct,np.matmul(P_pred,Ct.T)) + R)
        K_gain=np.matmul(P_pred,np.matmul(Ct.T,S))
        K_gains.append(K_gain.flatten())
        X_corrected = X_pred + np.matmul(K_gain,(Zt - Ct @ X_pred))
        P_corrected =np.matmul( (np.eye(2) - np.matmul(K_gain ,Ct)),P_pred)

        X = X_corrected
        P = P_corrected
        X_Predictions.append(X_pred.flatten())  
        P_Predictions.append(P_pred.diagonal())  
        X_Corrections.append(X_corrected.flatten())  
        P_Corrections.append(P_corrected.diagonal())  
        X_grounds.append(X_ground.flatten()) 

    X_Predictions = np.array(X_Predictions)
    X_Corrections = np.array(X_Corrections)
    X_grounds = np.array(X_grounds)
    P_Predictions = np.array(P_Predictions)
    P_Corrections = np.array(P_Corrections)
    K_gains=np.array(K_gains)

    sigma_x = np.sqrt(P_Predictions[:, 0]) 
    upper_bound_x = X_Corrections[:, 0] + sigma_x
    lower_bound_x = X_Corrections[:, 0] - sigma_x

    sigma_x_dot = np.sqrt(P_Predictions[:, 1]) 
    upper_bound_dot_x = X_Corrections[:, 1] + sigma_x_dot
    lower_bound_dot_x = X_Corrections[:, 1] - sigma_x_dot
    fig = go.Figure()
  
    fig.add_trace(go.Scatter(x=time_series, y=K_gains[:, 0], mode='lines', name="K_g", line=dict(dash="dot", color="red")))
    fig.add_trace(go.Scatter(x=time_series, y=K_gains[:, 1], mode='lines', name="dot_K_g", line=dict(color="red")))
    fig.update_layout(
        title="K_gains",
        xaxis_title="Time (s)",
        yaxis_title="K_gain & dot_K_gain",
        legend_title="Legend", 
    )
    fig.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_grounds[:, 0], mode='lines', name="Ground Truth Trajectory", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 0], mode='lines', name="Estimated Trajectory", line=dict(color="red")))
    fig.add_trace(go.Scatter(
        x=time_series, y=upper_bound_x, mode='lines',
        name="+1 Sigma", line=dict(color="rgba(255, 160, 160, 0.5)"), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=time_series, y=lower_bound_x, mode='lines',
        name="-1 Sigma", line=dict(color="rgba(255, 160, 160, 0.53)"), fill='tonexty', showlegend=False
    ))
    fig.update_layout(
        title="Trajectory with uncertainity",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        
    )
    fig.show()

    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_grounds[:, 0], mode='lines', name="Ground Truth Trajectory", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 0], mode='lines', name="Estimated Trajectory", line=dict(color="red")))
    
    fig.update_layout(
        title="Trajectory",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        
    )
    fig.show()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_grounds[:, 1], mode='lines', name="Ground Truth Velocity", line=dict(color="Green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 1], mode='lines', name="Estimated Velocity", line=dict(color="red")))
    fig.update_layout(
        title="Velocity",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        
    )
    fig.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_grounds[:, 1], mode='lines', name="Ground Truth Velocity", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 1], mode='lines', name="Estimated Velocity", line=dict(color="red")))
    fig.add_trace(go.Scatter(
        x=time_series, y=upper_bound_dot_x, mode='lines',
        name="+1 Sigma", line=dict(color="rgba(255, 160, 160, 0.5)"), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=time_series, y=lower_bound_dot_x, mode='lines',
        name="-1 Sigma", line=dict(color="rgba(255, 160, 160, 0.53)"), fill='tonexty', showlegend=False
    ))
    fig.update_layout(
        title="Velcoity with uncertainity",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        
    )
    fig.show()

def one_D_mech_fail():
    np.random.seed(32)
    del_t = 0.01 
    Vs = 3000  

    At = np.array([[1, del_t], [0, 1]]) 
    Bt = np.array([[0.5 * (del_t ** 2)], [del_t]])  
    Ct = np.array([[2 / Vs, 0]]) 

    X = np.array([[0], [0]]) 
    Q = np.diag([0.1**2, 0.5**2])
    P = np.array([[10**-4, 0], [0, 10**-4]]) 
    R = np.array([[0.01**2]])  

    def control_input(t):
        if t < 0.25:
            return np.array([[400]])
        elif 3.0 < t < 3.25:
            return np.array([[-400]])
        else:
            return np.array([[0]])
    t = 0
    time_series = []
    time_series = [0]
    K_gains=[[0,0]]
    X_Predictions = [X.flatten()]
    P_Predictions = [X.flatten()]
    X_Corrections = [X.flatten()]
    P_Corrections = [X.flatten()]
    X_grounds=[X.flatten()]
    X_ground=X
    for i in range(326):
        t += del_t 
        time_series.append(t)
        Ut = control_input(t)

        # Prediction
        motion_noise = np.random.multivariate_normal([0, 0], Q).reshape(2, 1)
        X_ground = np.matmul(At, X_ground) + np.matmul(Bt, Ut)+motion_noise

        measurement_noise = np.random.normal(0, np.sqrt(R), size=(1, 1)) 
        Zt = (2 * X_ground[0, 0]) / Vs + measurement_noise 
       
        X_pred=np.matmul(At,X)+np.matmul(Bt,Ut)
        P_pred=np.matmul(At,np.matmul(P,At.T))+Q
        if t<=2.5 and t>=1.5:
            X_corrected=X_pred
            P_corrected=P_pred
            X_Predictions.append(X_pred.flatten())  
            P_Predictions.append(P_pred.diagonal())  
            X_Corrections.append(X_corrected.flatten())  
            P_Corrections.append(P_corrected.diagonal())  
            X_grounds.append(X_ground.flatten()) 
            X = X_corrected
            P = P_corrected
            continue

        # Kalman gain
        S=np.linalg.inv(np.matmul(Ct,np.matmul(P_pred,Ct.T)) + R)
        K_gain=np.matmul(P_pred,np.matmul(Ct.T,S))
        K_gains.append(K_gain.flatten())
        X_corrected = X_pred + np.matmul(K_gain,(Zt - Ct @ X_pred))
        P_corrected =np.matmul( (np.eye(2) - np.matmul(K_gain ,Ct)),P_pred)

        X = X_corrected
        P = P_corrected
        X_Predictions.append(X_pred.flatten())  
        P_Predictions.append(P_pred.diagonal())  
        X_Corrections.append(X_corrected.flatten())  
        P_Corrections.append(P_corrected.diagonal())  
        X_grounds.append(X_ground.flatten()) 

    X_Predictions = np.array(X_Predictions)
    X_Corrections = np.array(X_Corrections)
    X_grounds = np.array(X_grounds)
    P_Predictions = np.array(P_Predictions)
    P_Corrections = np.array(P_Corrections)
    K_gains=np.array(K_gains)

    sigma_x = np.sqrt(P_Predictions[:, 0]) 
    upper_bound_x = X_Corrections[:, 0] + sigma_x
    lower_bound_x = X_Corrections[:, 0] - sigma_x

    sigma_x_dot = np.sqrt(P_Predictions[:, 1]) 
    upper_bound_dot_x = X_Corrections[:, 1] + sigma_x_dot
    lower_bound_dot_x = X_Corrections[:, 1] - sigma_x_dot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_grounds[:, 0], mode='lines', name="Ground Truth Trajectory", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 0], mode='lines', name="Estimated Trajectory", line=dict(color="red")))
    fig.add_trace(go.Scatter(
        x=time_series, y=upper_bound_x, mode='lines',
        name="+1 Sigma", line=dict(color="rgba(255, 160, 160, 0.5)"), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=time_series, y=lower_bound_x, mode='lines',
        name="-1 Sigma", line=dict(color="rgba(255, 160, 160, 0.53)"), fill='tonexty', showlegend=False
    ))
    fig.update_layout(
        title="Trajectory with uncertainity",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        
    )
    fig.show()

    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_grounds[:, 0], mode='lines', name="Ground Truth Trajectory", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 0], mode='lines', name="Estimated Trajectory", line=dict(color="red")))
    
    fig.update_layout(
        title="Trajectory",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        
    )
    fig.show()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_grounds[:, 1], mode='lines', name="Ground Truth Velocity", line=dict(color="Green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 1], mode='lines', name="Estimated Velocity", line=dict(color="red")))
    fig.update_layout(
        title="Velocity",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        
    )
    fig.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_grounds[:, 1], mode='lines', name="Ground Truth Velocity", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 1], mode='lines', name="Estimated Velocity", line=dict(color="red")))
    fig.add_trace(go.Scatter(
        x=time_series, y=upper_bound_dot_x, mode='lines',
        name="+1 Sigma", line=dict(color="rgba(255, 160, 160, 0.5)"), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=time_series, y=lower_bound_dot_x, mode='lines',
        name="-1 Sigma", line=dict(color="rgba(255, 160, 160, 0.53)"), fill='tonexty', showlegend=False
    ))
    fig.update_layout(
        title="Velcoity with uncertainity",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        
    )
    fig.show()
def ground():
    del_t = 0.01
    N = 130 
    g = -10 
    At = np.array([
        [1, 0, 0, del_t, 0, 0], 
        [0, 1, 0, 0, del_t, 0], 
        [0, 0, 1, 0, 0, del_t], 
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1]
    ])

    Bt = np.array([[0], [0], [0.5 * (del_t ** 2)], [0], [0], [del_t]]) 
    X_ground = np.array([[24.0], [4.0], [0.0], [-16.04], [36.8], [8.61]])
    time_series = []
    X_grounds=[]
    t = 0
    for _ in range(N):
        t += del_t
        time_series.append(t)
        Ut = np.array([[g]])  
        X_ground = np.matmul(At, X_ground) + np.matmul(Bt, Ut)
        X_grounds.append(X_ground.flatten())
    
    return X_grounds

def GPS():
    del_t = 0.01
    N = 130 
    g = -10 
    At = np.array([
        [1, 0, 0, del_t, 0, 0], 
        [0, 1, 0, 0, del_t, 0], 
        [0, 0, 1, 0, 0, del_t], 
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1]
    ])

    Bt = np.array([[0], [0], [0.5 * (del_t ** 2)], [0], [0], [del_t]]) 

    Ct_GPS = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    R_GPS = 0.01 * np.eye(3)  

    X = np.array([[24.0], [4.0], [0.0], [-16.04], [36.8], [8.61]])  
    P = 0.0001 * np.eye(6)
    Q = np.diag([0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01])


    time_series = [0]
    X_Predictions = [X.flatten()]
    P_Predictions = [X.flatten()]
    X_Corrections = [X.flatten()]
    P_Corrections = [X.flatten()]
    X_grounds=[X.flatten()]
    X_ground=X
    t = 0
    for _ in range(N):
        t += del_t
        time_series.append(t)
        motion_noise = np.random.multivariate_normal(np.zeros(6), Q).reshape(6, 1)
        Ut = np.array([[g]]) 
        X_ground=np.matmul(At, X) + np.matmul(Bt, Ut) + motion_noise
        X_pred = np.matmul(At, X) + np.matmul(Bt, Ut)
        P_pred = np.matmul(At, np.matmul(P, At.T)) + Q

        #Measurement
        measurement_noise = np.random.multivariate_normal(np.zeros(3), R_GPS).reshape(3, 1)
        Zt_GPS = np.matmul(Ct_GPS, X_ground) + measurement_noise

        #Kalman Gain
        S = np.matmul(Ct_GPS, np.matmul(P_pred, Ct_GPS.T)) + R_GPS
        K_gain_GPS = np.matmul(np.matmul(P_pred, Ct_GPS.T), np.linalg.inv(S))

        #Correction Step
        X_corrected = X_pred + np.matmul(K_gain_GPS, (Zt_GPS - np.matmul(Ct_GPS, X_pred)))
        P_corrected = np.matmul((np.eye(6) - np.matmul(K_gain_GPS, Ct_GPS)), P_pred)
        

        X_Predictions.append(X_pred.flatten())
        X_Corrections.append(X_corrected.flatten())
        X_grounds.append(X_ground.flatten()) 
        X = X_corrected
        P = P_corrected

    X_Predictions = np.vstack(X_Predictions)
    X_Corrections = np.vstack(X_Corrections)
    X_grounds=np.vstack(X_grounds)

    labels = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_grounds[:, 0], 
        y=X_grounds[:, 1], 
        z=X_grounds[:, 2], 
        mode='lines',
        name="Ground Truth Trajectory",
        line=dict(dash="solid", color="blue") 
    ))
    fig.update_layout(
        title="Ground Truth Trajectory",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )

    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_Corrections[:, 0], 
        y=X_Corrections[:, 1], 
        z=X_Corrections[:, 2], 
        mode='lines',
        name="Estimated Trajectory",
        line=dict(dash="solid", color="blue") 
    ))
    fig.update_layout(
        title="GPS 3D Estimated Trajectory",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )

    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 0], mode='lines', name="Estimated x_t", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 1], mode='lines', name="Estimated y_t", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 2], mode='lines', name="Estimated z_t", line=dict(color="orange")))
    fig.update_layout(
        title="GPS Estimated Trajectory",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        template="plotly_dark"
    )

    fig.show()

def IMU():
    del_t = 1 / 100 
    N=130
    At = np.array([
        [1, 0, 0, del_t, 0, 0], 
        [0, 1, 0, 0, del_t, 0], 
        [0, 0, 1, 0, 0, del_t], 
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1]
    ]) 
    Bt = np.array([[0], [0], [0.5 * del_t ** 2], [0], [0], [del_t]]) 
    Ct_IMU = np.eye(6) 
    R_IMU = 0.01 * np.eye(6) 

    x, y, z = 24, 4, 0
    d_x, d_y, d_z = -16.04, 36.8, 8.61
    X = np.array([[x], [y], [z], [d_x], [d_y], [d_z]]) 

    P = 0.0001 * np.eye(6) 
    Q = np.diag([0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01]) 

    t = 0
    time_series = [0]
    X_Predictions = [X.flatten()]
    P_Predictions = [X.flatten()]
    X_Corrections = [X.flatten()]
    P_Corrections = [X.flatten()]
    X_grounds=[X.flatten()]
    X_ground=X
    # Simulation Loop
    for _ in range(N):
        t += del_t 
        time_series.append(t)
        g = -10
        Ut = np.array([[g]])  

        #Prediction step
        motion_noise = np.random.multivariate_normal(np.zeros(6), Q).reshape(6, 1) 
        X_ground = np.matmul(At, X_ground) + np.matmul(Bt, Ut)+ motion_noise 
        X_pred = np.matmul(At, X) + np.matmul(Bt, Ut) 
        P_pred = np.matmul(At, np.matmul(P, At.T)) + Q 

        #Measurement
        measurement_noise = np.random.multivariate_normal(np.zeros(6), R_IMU).reshape(6, 1)  
        Zt_IMU = np.matmul(Ct_IMU, X_ground) + measurement_noise  

        #Kalman Gain
        S = np.matmul(Ct_IMU, np.matmul(P_pred, Ct_IMU.T)) + R_IMU
        K_gain_IMU = np.matmul(np.matmul(P_pred, Ct_IMU.T), np.linalg.inv(S))

        X_corrected = X_pred + np.matmul(K_gain_IMU, (Zt_IMU - np.matmul(Ct_IMU, X_pred)))
        P_corrected = np.matmul((np.eye(6) - np.matmul(K_gain_IMU, Ct_IMU)), P_pred)

        X = X_corrected
        P = P_corrected

        X_Predictions.append(X_pred.flatten())
        P_Predictions.append(P_pred.flatten())  
        X_Corrections.append(X_corrected.flatten())
        P_Corrections.append(P_corrected.flatten())
        X_grounds.append(X_ground.flatten()) 


    X_Predictions = np.vstack(X_Predictions)
    X_Corrections = np.vstack(X_Corrections)
    X_grounds=np.vstack(X_grounds)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_grounds[:, 0], 
        y=X_grounds[:, 1], 
        z=X_grounds[:, 2], 
        mode='lines',
        name="Groud Truth Trajectory",
        line=dict(dash="solid", color="blue") 
    ))

    fig.update_layout(
        title="Ground Truth Trajectory",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )

    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_Corrections[:, 0], 
        y=X_Corrections[:, 1], 
        z=X_Corrections[:, 2], 
        mode='lines',
        name="Estimated Trajectory",
        line=dict(dash="solid", color="blue") 
    ))

    fig.update_layout(
        title="IMU 3D Estimated Trajectory",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )

    fig.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 0], mode='lines', name="x_t", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 1], mode='lines', name="y_t", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 2], mode='lines', name="z_t", line=dict(color="orange")))
    fig.update_layout(
        title="IMU Estimated Trajectory",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        template="plotly_dark"
    )
    fig.show()

def BS():
    del_t = 0.01
    At = np.array([
        [1, 0, 0, del_t, 0, 0], 
        [0, 1, 0, 0, del_t, 0], 
        [0, 0, 1, 0, 0, del_t], 
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1]
    ]) 
    Bt = np.array([
        [0], [0], [0.5 * (del_t ** 2)], 
        [0], [0], [del_t] 
    ])  
    R_BS = 0.01 * np.eye(4)  
    X = np.array([[24], [4], [0], [-16.04], [36.8], [8.61]]) 
    P = 0.0001 * np.eye(6)  
    Q = np.diag([0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01]) 
    base_stations = np.array([
        [32, 50, 10],
        [-32, 50, 10],
        [-32, -50, 10],
        [32, -50, 10]
    ])
    t = 0
    time_series = [0]

    X_Predictions = [X.flatten()]
    P_Predictions = [X.flatten()]
    X_Corrections = [X.flatten()]
    P_Corrections = [X.flatten()]
    X_grounds=[X.flatten()]
    X_ground=X
    while t < 1.3:
        t += del_t  
        time_series.append(t)
        g = -10 
        Ut = np.array([[g]])  
        
        #Prediction step
        motion_noise = np.random.multivariate_normal(np.zeros(6), Q).reshape(6, 1)
        X_ground = np.matmul(At, X_ground) + np.matmul(Bt, Ut) + motion_noise 
        X_pred = np.matmul(At, X) + np.matmul(Bt, Ut) 
        P_pred = np.matmul(At, np.matmul(P, At.T)) + Q  

        D = np.linalg.norm(base_stations - X_ground[:3, 0], axis=1).reshape(4, 1)  
        H = np.zeros((4, 6))
        for i in range(4):
            diff = X[:3, 0] - base_stations[i]  
            norm = np.linalg.norm(diff)  
            H[i, :3] = diff / norm 
       
        measurement_noise = np.random.multivariate_normal(np.zeros(4), R_BS).reshape(4, 1)
        Zt_BS = D + measurement_noise  

        S = np.matmul(H, np.matmul(P_pred, H.T)) + R_BS
        K_gain_BS = np.matmul(np.matmul(P_pred, H.T), np.linalg.inv(S))

        #Correction Step
        X_corrected = X_pred + np.matmul(K_gain_BS, (Zt_BS - D))  
        P_corrected = np.matmul((np.eye(6) - np.matmul(K_gain_BS, H)), P_pred)
        
        #Update
        X = X_corrected
        P = P_corrected

        X_Predictions.append(X_pred.flatten())
        P_Predictions.append(P_pred.flatten())  
        X_Corrections.append(X_corrected.flatten())
        P_Corrections.append(P_corrected.flatten())
        X_grounds.append(X_ground.flatten())

    X_Predictions = np.vstack(X_Predictions)
    X_Corrections = np.vstack(X_Corrections)
    X_grounds = np.vstack(X_grounds)
    

    fig = go.Figure()

    #Plod 3D trajectory
    fig.add_trace(go.Scatter3d(
        x=X_grounds[:, 0], 
        y=X_grounds[:, 1], 
        z=X_grounds[:, 2], 
        mode='lines',
        name="Estimated Trajectory",
        line=dict(dash="solid", color="blue")  
    ))

    fig.update_layout(
        title="Ground Truth Trajectory",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    fig.show()
    fig = go.Figure()

    #Plod 3D trajectory
    fig.add_trace(go.Scatter3d(
        x=X_Corrections[:, 0], 
        y=X_Corrections[:, 1], 
        z=X_Corrections[:, 2], 
        mode='lines',
        name="Estimated Trajectory",
        line=dict(dash="solid", color="blue")  
    ))

    fig.update_layout(
        title="Base Stations 3D Estimated Trajectory",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    fig.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 0], mode='lines', name="x_t", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 1], mode='lines', name="y_t", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=time_series, y=X_Corrections[:, 2], mode='lines', name="z_t", line=dict(color="orange")))

    fig.update_layout(
        title="Base Stations Estimated Trajectory",
        xaxis_title="Time (s)",
        yaxis_title="State Variables",
        legend_title="Legend",
        template="plotly_dark"
    )
    fig.show()
def sim_GPS():
    n=0
    h=0
    p=0
    np.random.seed(42)
    for _ in range(1000):
       
        del_t = 0.01 
        N = 130  
        g = -10 
        At = np.array([
            [1, 0, 0, del_t, 0, 0], 
            [0, 1, 0, 0, del_t, 0], 
            [0, 0, 1, 0, 0, del_t], 
            [0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 1]
        ])

        Bt = np.array([[0], [0], [0.5 * (del_t ** 2)], [0], [0], [del_t]])  

        Ct_GPS = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        R_GPS = 0.01 * np.eye(3) 
        # R_GPS = (0.05**2) * np.eye(3) 
        X = np.array([[24.0], [4.0], [0.0], [-16.04], [36.8], [8.61]]) 
        P = 0.0001 * np.eye(6)  
        Q = np.diag([0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01])
        # Q = np.diag([0.005**2, 0.005**2, 0.005**2,0.05**2, 0.05**2, 0.05**2])
        t = 0
        time_series = []
        X_Predictions = []
        P_Predictions = []
        X_Corrections = []
        P_Corrections = []
        Z_observations=[]
        X_grounds=[]
        X_ground=X
        for _ in range(N):
            t += del_t
            time_series.append(t)
            Ut = np.array([[g]]) 

            #Prediction Step
            motion_noise = np.random.multivariate_normal(np.zeros(6), Q).reshape(6, 1)
            X_ground = np.matmul(At, X_ground) + np.matmul(Bt, Ut)+ motion_noise
            X_pred = np.matmul(At, X) + np.matmul(Bt, Ut) 
            P_pred = np.matmul(At, np.matmul(P, At.T)) + Q

            #Measurement
            measurement_noise = np.random.multivariate_normal(np.zeros(3), R_GPS).reshape(3, 1)
            Zt_GPS = np.matmul(Ct_GPS, X_ground) + measurement_noise

            #Kalman Gain
            S = np.matmul(Ct_GPS, np.matmul(P_pred, Ct_GPS.T)) + R_GPS
            K_gain_GPS = np.matmul(np.matmul(P_pred, Ct_GPS.T), np.linalg.inv(S))

            #Correction Step
            X_corrected = X_pred + np.matmul(K_gain_GPS, (Zt_GPS - np.matmul(Ct_GPS, X_pred)))
            P_corrected = np.matmul((np.eye(6) - np.matmul(K_gain_GPS, Ct_GPS)), P_pred)


            X = X_corrected
            P = P_corrected
            X_Predictions.append(X_pred)
            X_Corrections.append(X_corrected)
            Z_observations.append(Zt_GPS)
            X_grounds.append(X_ground)
        #Automated referee for declaring if a goal is scored.
        for x_g in X_grounds:
            if  x_g[1][0]>=50:
                if x_g[0][0]<=4 and x_g[2][0]<=3:
                    p+=1
                break
        for x in X_Corrections:
            if  x[1][0]>=50 :
                if x[0][0]<=4 and x[2][0]<=3:
                    n+=1
                break
        for obs in Z_observations:
            if obs[0][0]<=4  and obs[1][0]>=50  and obs[2][0]<=3 :
                if obs[0][0]<=4 and obs[2][0]<=3:
                    h+=1
                break
    print("Number of goals scored by using Ground Truth Trajectory:",p)
    print("Number of goals scored by using Estimated Trajectory:",n)
    print("Number of goals scored by using GPS Measuremnts:",h)

def sim_IMU():
    n=0
    h=0
    p=0
    np.random.seed(42)
    for i in range(1000):
        del_t = 1 / 100 
        N=130
        At = np.array([
            [1, 0, 0, del_t, 0, 0], 
            [0, 1, 0, 0, del_t, 0], 
            [0, 0, 1, 0, 0, del_t], 
            [0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 1]
        ]) 

        Bt = np.array([[0], [0], [0.5 * del_t ** 2], [0], [0], [del_t]])  
        Ct_IMU = np.eye(6) 
        R_IMU = 0.01 * np.eye(6) 
        x, y, z = 24, 4, 0
        d_x, d_y, d_z = -16.04, 36.8, 8.61
        X = np.array([[x], [y], [z], [d_x], [d_y], [d_z]])  
        P = 0.0001 * np.eye(6) 
        Q = np.diag([0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01]) 

        t = 0
        time_series = []
        X_Predictions = []
        P_Predictions = []
        X_Corrections = []
        P_Corrections = []
        Z_observations=[]
        X_grounds=[]
        X_ground=X

        for _ in range(N):
            t += del_t 
            time_series.append(t)
            g = -10
            Ut = np.array([[g]])
 
            motion_noise = np.random.multivariate_normal(np.zeros(6), Q).reshape(6, 1) 
            X_ground = np.matmul(At, X_ground) + np.matmul(Bt, Ut)  + motion_noise 
            X_pred = np.matmul(At, X) + np.matmul(Bt, Ut)
            P_pred = np.matmul(At, np.matmul(P, At.T)) + Q 

            measurement_noise = np.random.multivariate_normal(np.zeros(6), R_IMU).reshape(6, 1) 
            Zt_IMU = np.matmul(Ct_IMU, X_ground) + measurement_noise  

            S = np.matmul(Ct_IMU, np.matmul(P_pred, Ct_IMU.T)) + R_IMU
            K_gain_IMU = np.matmul(np.matmul(P_pred, Ct_IMU.T), np.linalg.inv(S))

            X_corrected = X_pred + np.matmul(K_gain_IMU, (Zt_IMU - np.matmul(Ct_IMU, X_pred)))
            P_corrected = np.matmul((np.eye(6) - np.matmul(K_gain_IMU, Ct_IMU)), P_pred)

            X = X_corrected
            P = P_corrected

            X_Predictions.append(X_pred)
            X_Corrections.append(X_corrected)
            Z_observations.append(Zt_IMU)
            X_grounds.append(X_ground)

        #Automated referee for declaring if a goal is scored.
        for x_g in X_grounds:
            if  x_g[1][0]>=50:
                if x_g[0][0]<=4 and x_g[2][0]<=3:
                    p+=1
                break
        for x in X_Corrections:
            if  x[1][0]>=50 :
                if x[0][0]<=4 and x[2][0]<=3:
                    n+=1
                break
        for obs in Z_observations:
            if obs[0][0]<=4  and obs[1][0]>=50  and obs[2][0]<=3 :
                if obs[0][0]<=4 and obs[2][0]<=3:
                    h+=1
                break
    print("Number of goals scored by using Ground Truth:",p)
    print("Number of goals scored by using Estimated Trajectory:",n)
    print("Number of goals scored by using IMU Measuremnts:",h)


def GPS_ellipse():
    del_t = 0.01 
    N = 130 
    g = -10 
    At = np.array([
        [1, 0, 0, del_t, 0, 0], 
        [0, 1, 0, 0, del_t, 0], 
        [0, 0, 1, 0, 0, del_t], 
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1]
    ])

    Bt = np.array([[0], [0], [0.5 * (del_t ** 2)], [0], [0], [del_t]]) 
    Ct_GPS = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    R_GPS = 0.01 * np.eye(3) 
    X = np.array([[24.0], [4.0], [0.0], [-16.04], [36.8], [8.61]]) 
    P = 0.0001 * np.eye(6)
    X_ground=X
    Q = np.diag([0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01])

    time_series = []
    X_Predictions = []
    X_Corrections = []
    X_grounds=[]
    uncertainty_ellipses=[]

    t = 0
    X_ground=X
    for _ in range(N):
        t += del_t
        time_series.append(t)
        Ut = np.array([[g]]) 

        motion_noise = np.random.multivariate_normal(np.zeros(6), Q).reshape(6, 1)
        X_ground = np.matmul(At, X_ground) + np.matmul(Bt, Ut)+ motion_noise
        X_pred = np.matmul(At, X) + np.matmul(Bt, Ut) 
        P_pred = np.matmul(At, np.matmul(P, At.T)) + Q

        measurement_noise = np.random.multivariate_normal(np.zeros(3), R_GPS).reshape(3, 1)
        Zt_GPS = np.matmul(Ct_GPS, X_ground) + measurement_noise

        S = np.matmul(Ct_GPS, np.matmul(P_pred, Ct_GPS.T)) + R_GPS
        K_gain_GPS = np.matmul(np.matmul(P_pred, Ct_GPS.T), np.linalg.inv(S))

        X_corrected = X_pred + np.matmul(K_gain_GPS, (Zt_GPS - np.matmul(Ct_GPS, X_pred)))
        P_corrected = np.matmul((np.eye(6) - np.matmul(K_gain_GPS, Ct_GPS)), P_pred)
        uncertainty_ellipses.append((X_corrected[0, 0], X_corrected[1, 0], P_corrected[:2, :2]))

        X_Predictions.append(X_pred.flatten())
        X_Corrections.append(X_corrected.flatten())
        X_grounds.append(X_ground.flatten())
        X = X_corrected
        P = P_corrected

    X_Predictions = np.vstack(X_Predictions)
    X_Corrections = np.vstack(X_Corrections)
    X_grounds=np.vstack(X_grounds)

    def generate_ellipse(mean_x, mean_y, cov_matrix, num_points=100):
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        angle = np.arctan2(*eigenvecs[:, 0][::-1])
        
        theta = np.linspace(0, 2 * np.pi, num_points)
        ellipse_x = np.sqrt(eigenvals[0]) * np.cos(theta)
        ellipse_y = np.sqrt(eigenvals[1]) * np.sin(theta)

        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ellipse_points = np.dot(R, np.array([ellipse_x, ellipse_y]))
        
        return ellipse_points[0] + mean_x, ellipse_points[1] + mean_y

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_Corrections[:, 0], 
        y=X_Corrections[:, 1], 
        mode='lines', 
        name='Estimated Trajectory', 
        line=dict(color="green")
    ))

    # Plot uncertainty ellipses
    for x, y, P_xy in uncertainty_ellipses:
        ellipse_x, ellipse_y = generate_ellipse(x, y, P_xy)
        fig.add_trace(go.Scatter(
            x=ellipse_x, y=ellipse_y, mode='lines', 
            line=dict(color="rgba(255, 0, 0, 0.5)"),  
            showlegend=False
        ))

    fig.update_layout(
        title="XY Projection with Uncertainty Ellipses",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        legend_title="Legend",
        template="plotly_white"
    )

    fig.show()

def Bs_2D():

    np.random.seed(32)
    del_t = 0.01
    At = np.array([
        [1, 0, del_t, 0],
        [0, 1, 0, del_t],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    X_init = np.array([[0.0], [-50.0], [0.0], [40.0]])
    P_init = np.eye(4)
    Q = np.diag([0.0001, 0.0001, 0.01, 0.01])
    base_stations = {
        "B1": np.array([-32, 50]),
        "B2": np.array([-32, -50]),
        "B3": np.array([-32, 0]),
        "B4": np.array([32, 0])
    }
    observation_cases = [["B1"], ["B2"], ["B3"], ["B3", "B4"]]
    R_BS = 0.01
    timesteps = 250

    # Store
    trajectories = {}
    major_axes = {}
    minor_axes = {}
    uncertainty_ellipses = {}

    for case in observation_cases:
        case_str = str(case)
        base_stations_used = np.array([base_stations[bs] for bs in case])
        X = X_init.copy()
        P = P_init.copy()
        X_ground=X
        X_estimates = []
        major_lengths = []
        minor_lengths = []
        ellipse_data = []

        for i in range(timesteps):

            X_ground=np.matmul(At, X) + np.random.multivariate_normal(np.zeros(4), Q).reshape(4, 1)

            X_pred = np.matmul(At, X) 
            P_pred = np.matmul(np.matmul(At, P), At.T) + Q

            D = np.linalg.norm(base_stations_used - X_ground[:2, 0], axis=1).reshape(len(case), 1)
            H = np.zeros((len(case), 4))
            for i in range(len(case)):
                diff = X[:2, 0] - base_stations_used[i]
                norm = np.linalg.norm(diff)
                H[i, :2] = diff / norm 

            S = np.matmul(np.matmul(H, P_pred), H.T) + R_BS * np.eye(len(case))
            K = np.matmul(np.matmul(P_pred, H.T), np.linalg.inv(S))
            Z = D + np.random.normal(0, np.sqrt(R_BS), (len(case), 1))

            X = X_pred + np.matmul(K, (Z - D))
            P = np.matmul((np.eye(4) - np.matmul(K, H)), P_pred)

            X_estimates.append(X[:2, 0])
            eigenvals, eigenvecs = np.linalg.eigh(P[:2, :2])
            major_lengths.append(2 * np.sqrt(max(eigenvals)))
            minor_lengths.append(2 * np.sqrt(min(eigenvals)))

            if i% 10== 0:
                ellipse_data.append((X[:2, 0], eigenvals, eigenvecs))

        trajectories[case_str] = np.array(X_estimates)
        major_axes[case_str] = major_lengths
        minor_axes[case_str] = minor_lengths
        uncertainty_ellipses[case_str] = ellipse_data

    time_series = np.arange(timesteps) * del_t

    for case in observation_cases:
        case_str = str(case)
        traj = trajectories[case_str]
        base_stations_used = np.array([base_stations[bs] for bs in case])
        ellipses = uncertainty_ellipses[case_str]

        fig_traj = go.Figure()

        fig_traj.add_trace(go.Scatter(
            x=traj[:, 0], y=traj[:, 1],
            mode='lines', name=f'Trajectory {case_str}',
            line=dict(width=2, color='red')
        ))

        fig_traj.add_trace(go.Scatter(
            x=base_stations_used[:, 0], y=base_stations_used[:, 1],
            mode='markers', name='Base Stations',
            marker=dict(size=10, symbol='circle', color='green')
        ))

        for center, eigenvals, eigenvecs in ellipses:
            theta = np.linspace(0, 2 * np.pi, 100)
            ellipse_x = np.sqrt(eigenvals[1]) * np.cos(theta)
            ellipse_y = np.sqrt(eigenvals[0]) * np.sin(theta)
            ellipse = np.vstack((ellipse_x, ellipse_y))

            ellipse_rotated = np.dot(eigenvecs, ellipse)

            ellipse_rotated[0, :] += center[0]
            ellipse_rotated[1, :] += center[1]

            fig_traj.add_trace(go.Scatter(
                x=ellipse_rotated[0, :], y=ellipse_rotated[1, :],
                mode='lines', line=dict(width=1, color='blue'),
                showlegend=False
            ))

        fig_traj.update_layout(
            title=f"Estimated Trajectory with Uncertainty Ellipses - Case {case_str}",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            legend_title="Observation Case",
        
        )

        fig_traj.show()

    fig_uncertainty = go.Figure()

    for case in observation_cases:
        case_str = str(case)
        fig_uncertainty.add_trace(go.Scatter(
            x=time_series, y=major_axes[case_str],
            mode='lines', name=f'Major Axis {case_str}',
            line=dict(width=2)
        ))
        fig_uncertainty.add_trace(go.Scatter(
            x=time_series, y=minor_axes[case_str],
            mode='lines', name=f'Minor Axis {case_str}',
            line=dict(dash='dot')
        ))

    fig_uncertainty.update_layout(
        title="Major and Minor Axis Lengths of Uncertainty Ellipses vs Time",
        xaxis_title="Time (s)",
        yaxis_title="Axis Length",
        legend_title="Observation Cases",
    )

    fig_uncertainty.show()

def select_mode(mode):
    if mode=="1D":
        one_D()
    elif mode=="mech_fail":
        one_D_mech_fail()
    elif mode=="GPS":
        GPS()
    elif mode=="IMU":
        IMU()
    elif mode=="BS":
        BS()
    elif mode=="SIM_GPS":
        sim_GPS()
    elif mode=="SIM_IMU":
        sim_IMU()
    elif mode=="GPS_ELLIPSE":
        GPS_ellipse()
    elif mode=="2D_BS":
        Bs_2D()
    else:
        print("modes:1D,mech_fail, GPS, IMU, BS, SIM_GPS, SIM_IMU, GPS_ELLIPSE, 2D_BS")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 A1.py <mode>, modes:1D,mech_fail, GPS, IMU, BS, SIM_GPS, SIM_IMU, GPS_ELLIPSE, 2D_BS")
        sys.exit(1)

    mode = sys.argv[1]
    select_mode(mode)



