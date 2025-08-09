"""
CPG in polar coordinates based on: 
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import numpy as np

MASS = 12.454
GRAVITY = 9.81
FOOT_Y = 0.0838
foot_y = FOOT_Y
foot_x = 0

class HopfNetwork:
    """
    CPG network based on Hopf polar equations mapped to foot positions in Cartesian space.
    Foot Order is FR, FL, RR, RL (Front Right, Front Left, Rear Right, Rear Left)
    """
    def __init__(self,
                 mu=1.0,
                 omega_swing=None,
                 omega_stance=None,
                 gait="WALK",
                 coupling_strength=None,
                 couple=True,
                 time_step=0.001,
                 ground_clearance=None,
                 ground_penetration=None,
                 robot_height=None,
                 des_step_len=None):
        self.X = np.zeros((2, 4))
        self._mu = mu
        self._couple = couple
        self._dt = time_step
        self._set_gait(gait)
        self._coupling_strength = min(self._omega_swing, self._omega_stance) / 3 * np.ones((4, 4))
        if omega_stance is not None:
            self._omega_stance = omega_stance
        if omega_swing is not None:
            self._omega_swing = omega_swing
        if ground_clearance is not None:
            self._ground_clearance = ground_clearance
        if ground_penetration is not None:
            self._ground_penetration = ground_penetration
        if robot_height is not None:
            self._robot_height = robot_height
        if des_step_len is not None:
            self._des_step_len = des_step_len
        self.X[0, :] = np.ones((1, 4)) * 0.1
        self.X[1, :] = self.PHI[0, :]
        self.X_dot = np.zeros((2, 4))
        self.state = ["SWING"] * 4


    def _set_gait(self, gait):
        """ For coupling oscillators in phase space. [tODO] update all coupling matrices """
        # FL(2)  FR(1)
        # RL(4)  RR(3)
        def footfall2phi(footfall):
            ret = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    ret[i, j] = footfall[j] - footfall[i]
            return 2 * np.pi * ret
        footfall_trot = [0.5, 0.0, 0.0, 0.5]
        self.PHI_trot = footfall2phi(footfall_trot)
        footfall_walk = [0.0, 0.5, 0.75, 0.25]
        self.PHI_walk = footfall2phi(footfall_walk)
        footfall_walk_diagonal = [0.0, 0.5, 0.25, 0.75]
        self.PHI_walk_diagonal = footfall2phi(footfall_walk_diagonal)
        footfall_bound = [0.0, 0.0, 0.5, 0.5]
        self.PHI_bound = footfall2phi(footfall_bound)
        canter_param = 0.05
        footfall_canter_transverse = [0.5, 0.75 + canter_param, 0.25 - canter_param, 0.5]
        self.PHI_canter_transverse = footfall2phi(footfall_canter_transverse)
        footfall_canter_rotatory = [0.75 + canter_param, 0.5, 0.25 - canter_param, 0.5]
        self.PHI_canter_rotatory = footfall2phi(footfall_canter_rotatory)
        gallop_param1 = 0.05
        gallop_param2 = 0.25
        if gait == "BOUND":
            gallop_param1 = 0
            gallop_param2 = 0.25
        elif gait == "GALLOP_TRANS" or gait == "GALLOP":
            gallop_param2 = 0.4
        footfall_gallop_transverse = [gallop_param2 + gallop_param1, gallop_param2, gallop_param1, 0]
        self.PHI_gallop_transverse = footfall2phi(footfall_gallop_transverse)
        footfall_gallop_rotatory = [2 * gallop_param2 + gallop_param1, 2 * gallop_param2, 0, gallop_param1]
        self.PHI_gallop_rotatory = footfall2phi(footfall_gallop_rotatory)
        footfall_pace = [0.5, 0, 0.5, 0]
        self.PHI_pace = footfall2phi(footfall_pace)
        self.PHI_pronk = np.zeros((4, 4))
        self._des_step_len = 0.04
        self._ground_penetration = 0.01
        self._ground_clearance = 0.05
        self._robot_height = 0.25
        global foot_y, foot_x
        if gait == "WALK":
            print('WALK')
            self.PHI = self.PHI_walk
            self._omega_swing = 5.0 * 2 * np.pi
            self._omega_stance = 1.0 * 2 * np.pi
        elif gait == "WALK_DIAG":
            print('WALK_DIAG')
            self.PHI = self.PHI_walk_diagonal
            self._omega_swing = 5.0 * 2 * np.pi
            self._omega_stance = 3.0 * 2 * np.pi
        elif gait == "AMBLE":
            print(gait)
            self.PHI = self.PHI_walk
            self._omega_swing = 10.0 * 2 * np.pi
            self._omega_stance = 15.0 * 2 * np.pi
        elif gait == "TROT" or gait == "TROT_RUN":
            print('TROT_RUN')
            self.PHI = self.PHI_trot
            self._omega_swing = 9.0 * 2 * np.pi
            self._omega_stance = 10 * 2 * np.pi
        elif gait == "TROT_WALK":
            print('TROT_WALK')
            self.PHI = self.PHI_trot
            self._omega_swing = 2.2 * 2 * np.pi
            self._omega_stance = 2.0 * 2 * np.pi
        elif gait == "PACE":
            print('PACE')
            self.PHI = self.PHI_pace
            foot_y = FOOT_Y * 1.2
            self._robot_height *= 0.8
            self._omega_swing = 8.0 * 2 * np.pi
            self._omega_stance = 6.0 * 2 * np.pi
            self._des_step_len = 0.05
            self._ground_penetration = 0.01
            self._ground_clearance = 0.06
        elif gait == "PACE_FLY":
            print('PACE_FLY')
            self.PHI = self.PHI_pace
            foot_y = FOOT_Y * 1.5
            self._robot_height *= 0.8
            self._omega_swing = 6.0 * 2 * np.pi
            self._omega_stance = 9.0 * 2 * np.pi
            self._des_step_len = 0.05
        elif gait == "CANTER_TRANS" or gait == "CANTER":
            print("CANTER_TRANS")
            self.PHI = self.PHI_canter_transverse
            foot_y = FOOT_Y * 1.2
            self._robot_height *= 0.8
            self._omega_swing = 6 * 2 * np.pi
            self._omega_stance = 13 * 2 * np.pi
            self._des_step_len = 0.07
            self._ground_penetration = 0.005
            self._ground_clearance = 0.03
        elif gait == "CANTER_ROTA":
            print("CANTER_ROTA")
            self.PHI = self.PHI_canter_rotatory
            foot_y = FOOT_Y * 1.2
            self._robot_height *= 0.8
            self._omega_swing = 6.0 * 2 * np.pi
            self._omega_stance = 13.0 * 2 * np.pi
            self._des_step_len = 0.07
            self._ground_penetration = 0.005
            self._ground_clearance = 0.06
        elif gait == "BOUND":
            print(gait)
            self.PHI = self.PHI_gallop_rotatory
            foot_x = 0.04
            foot_y = FOOT_Y * 1.2
            self._robot_height = 0.20
            self._omega_swing = 5.5 * 2 * np.pi
            self._omega_stance = 22.0 * 2 * np.pi
            self._des_step_len = 0.065
            self._ground_penetration = 0.02
            self._ground_clearance = 0.07
        elif gait == "GALLOP_ROTA":
            print(gait)
            self.PHI = self.PHI_gallop_rotatory
            self._omega_swing = 11 * 2 * np.pi
            self._omega_stance = 20.0 * 2 * np.pi
            self._des_step_len = 0.05
            self._ground_penetration = 0.02
            self._ground_clearance = 0.05
            foot_y = FOOT_Y * 1.2
            self._robot_height = 0.23
            self._omega_swing = 9.5 * 2 * np.pi
            self._omega_stance = 20.0 * 2 * np.pi
            self._des_step_len = 0.05
            self._ground_penetration = 0.02
            self._ground_clearance = 0.06
        elif gait == "GALLOP_TRANS" or gait == "GALLOP":
            print(gait)
            self.PHI = self.PHI_gallop_transverse
            self._omega_swing = 10 * 2 * np.pi
            self._omega_stance = 15.0 * 2 * np.pi
            foot_y = FOOT_Y * 1.2
            self._robot_height *= 0.9
            self._omega_swing = 12 * 2 * np.pi
            self._omega_stance = 20.0 * 2 * np.pi
            self._des_step_len = 0.05
            self._ground_penetration = 0.02
        elif gait == "PRONK":
            print('PRONK')
            self.PHI = self.PHI_pronk
            self._robot_height = 0.20
            self._omega_swing = 11 * 2 * np.pi
            self._omega_stance = 25.0 * 2 * np.pi
            self._des_step_len = 0.05
            self._ground_penetration = 0.01
            self._ground_clearance = 0.07
            self._robot_height = 0.20
            foot_y = FOOT_Y * 1.2
            foot_x = 0.02
            self._omega_swing = 11 * 2 * np.pi
            self._omega_stance = 25.0 * 2 * np.pi
            self._des_step_len = 0.065
            self._ground_penetration = 0.022
            self._ground_clearance = 0.07
        else:
            raise ValueError(gait + ' not implemented.')

    def update(self, body_ori=None, contactInfo=None):
        """Update oscillator states and map to Cartesian foot positions."""
        self._integrate_hopf_equations(body_ori, contactInfo)
        r, theta = self.X[0, :], self.X[1, :]
        x = -self._des_step_len * r * np.cos(theta)
        z = np.zeros(4)
        for i in range(4):
            if theta[i] < np.pi:
                g = self._ground_clearance
                self.state[i] = "SWING"
            else:
                g = self._ground_penetration
                self.state[i] = "STANCE"
            z[i] = -self._robot_height + r[i] * np.sin(theta[i]) * g
        return x, z
      
        
    def _integrate_hopf_equations(self, body_ori=None, contactInfo=None):
        """Hopf polar equations and integration. Use equations 6 and 7."""
        X = self.X.copy()
        self.X_dot = np.zeros((2, 4))
        alpha = 5.0
        force_threshold = 10
        for i in range(4):
            r, theta = X[0, i], X[1, i]
            r_dot = alpha * (self._mu - r ** 2) * r
            if theta < np.pi:
                theta_dot = self._omega_swing
            else:
                theta_dot = self._omega_stance
            if self._couple:
                for j in range(4):
                    delta_theta = (X[1, j] - theta - self.PHI[i, j])
                    theta_dot += X[0, j] * self._coupling_strength[i, j] * np.sin(delta_theta)
            if theta_dot < 0:
                theta_dot = 0
            if contactInfo is not None:
                contactBool, forceNormal = contactInfo
                suppose_theta = theta + theta_dot * self._dt
                if theta < np.pi:
                    if suppose_theta > np.pi and not contactBool[i]:
                        theta_dot = 0
                    elif contactBool[i]:
                        theta_dot *= 2.0
                else:
                    if suppose_theta > 2 * np.pi and (forceNormal[i] > force_threshold):
                        theta_dot = 0
                    elif forceNormal[i] < force_threshold:
                        theta_dot *= 2.0
            self.X_dot[:, i] = [r_dot, theta_dot]
        self.X += self.X_dot * self._dt
        self.X[1, :] = self.X[1, :] % (2 * np.pi)


if __name__ == "__main__":
    # --- Kinematics and Dynamics Utilities ---
    def compute_foot_fk(joint_angles_leg):
        """
        Forward kinematics for Unitree Go1 leg (3-DOF).
        Args:
            joint_angles_leg (np.ndarray): [hip_abduction, hip_flexion, knee_flexion]
        Returns:
            np.ndarray: foot position (x, y, z) in leg base frame
        """
        # Go1 leg link lengths (meters)
        L1 = 0.085  # hip to thigh
        L2 = 0.2    # thigh to calf
        L3 = 0.2    # calf to foot
        q1, q2, q3 = joint_angles_leg
        # Hip abduction rotates about Y axis (outward)
        # Hip/knee flexion are planar (XZ)
        # Compute planar position first
        x = L2 * np.sin(q2) + L3 * np.sin(q2 + q3)
        z = - (L2 * np.cos(q2) + L3 * np.cos(q2 + q3))
        # Abduction moves in Y
        y = L1 * np.sin(q1)
        # Project foot position by abduction
        foot_pos = np.array([
            x * np.cos(q1),
            y,
            z
        ])
        return foot_pos

    def compute_leg_jacobian(joint_angles_leg):
        """
        Analytical Jacobian for Unitree Go1 leg (3-DOF).
        Args:
            joint_angles_leg (np.ndarray): [hip_abduction, hip_flexion, knee_flexion]
        Returns:
            np.ndarray: Jacobian matrix (3x3)
        """
        L1 = 0.085
        L2 = 0.2
        L3 = 0.2
        q1, q2, q3 = joint_angles_leg
        # Partial derivatives
        # dX/dq1, dY/dq1, dZ/dq1
        dx_dq1 = - (L2 * np.sin(q2) + L3 * np.sin(q2 + q3)) * np.sin(q1)
        dy_dq1 = L1 * np.cos(q1)
        dz_dq1 = 0
        # dX/dq2, dY/dq2, dZ/dq2
        dx_dq2 = L2 * np.cos(q2) * np.cos(q1) + L3 * np.cos(q2 + q3) * np.cos(q1)
        dy_dq2 = 0
        dz_dq2 = L2 * np.sin(q2) + L3 * np.sin(q2 + q3)
        # dX/dq3, dY/dq3, dZ/dq3
        dx_dq3 = L3 * np.cos(q2 + q3) * np.cos(q1)
        dy_dq3 = 0
        dz_dq3 = L3 * np.sin(q2 + q3)
        J = np.array([
            [dx_dq1, dx_dq2, dx_dq3],
            [dy_dq1, dy_dq2, dy_dq3],
            [dz_dq1, dz_dq2, dz_dq3]
        ])
        return J

    def compute_foot_velocity(joint_angles_leg, joint_velocities_leg):
        """
        Compute foot velocity using Jacobian and joint velocities.
        Args:
            joint_angles_leg (np.ndarray): shape (3,)
            joint_velocities_leg (np.ndarray): shape (3,)
        Returns:
            np.ndarray: foot velocity (vx, vy, vz)
        """
        J = compute_leg_jacobian(joint_angles_leg)
        return J @ joint_velocities_leg

    def compute_leg_ik(foot_xyz):
        """
        Closed-form IK for Unitree Go1 leg (3-DOF).
        Args:
            foot_xyz (np.ndarray): desired foot position (x, y, z)
        Returns:
            np.ndarray: joint angles (3,)
        """
        L1 = 0.085
        L2 = 0.2
        L3 = 0.2
        x, y, z = foot_xyz
        # Hip abduction (q1) from y
        if abs(y) > L1:
            y = np.clip(y, -L1, L1)
        q1 = np.arcsin(y / L1)
        # Project to sagittal plane
        xp = x / np.cos(q1)
        zp = z
        # Planar 2R IK for q2, q3
        D = (xp**2 + zp**2 - L2**2 - L3**2) / (2 * L2 * L3)
        D = np.clip(D, -1.0, 1.0)
        q3 = np.arccos(D)
        # Elbow-down solution
        phi = np.arctan2(zp, xp)
        psi = np.arctan2(L3 * np.sin(q3), L2 + L3 * np.cos(q3))
        q2 = phi - psi
        return np.array([q1, q2, q3])

    def compute_energy_cost(joint_torques, joint_velocities):
        """
        Compute energy cost for the robot.
        Args:
            joint_torques (np.ndarray): shape (12,)
            joint_velocities (np.ndarray): shape (12,)
        Returns:
            float: energy cost
        """
        # Example: sum of absolute mechanical work
        return np.sum(np.abs(joint_torques * joint_velocities))
    import time
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from gazebo_msgs.msg import ContactsState
    from nav_msgs.msg import Odometry
    import threading

    ADD_CARTESIAN_PD = True
    ADD_JOINT_PD = True
    USE_FEEDBACK = False
    TRACK_DIRECTION = False
    gait_direction = 0
    gait_name = "PRONK"
    simulation_time = 10.0
    PLAY_SPEED = 0.25
    notpureforward = True if gait_direction != 0.0 else False
    TIME_STEP = 0.001
    sideSign = np.array([-1, 1, -1, 1])

    # Data holders for Gazebo feedback
    joint_angles = np.zeros(12)
    joint_velocities = np.zeros(12)
    base_orientation = np.zeros(3)
    feet_contacts = np.zeros(4, dtype=bool)
    feet_forces = np.zeros(4)
    base_linear_velocity = np.zeros(3)
    base_position = np.zeros(3)

    # ROS 2 node for feedback
    class GazeboFeedback(Node):
        def __init__(self):
            super().__init__('gazebo_feedback')
            self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
            self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
            self.contacts_sub = self.create_subscription(ContactsState, '/foot_contact_states', self.contacts_callback, 10)
            self.lock = threading.Lock()

        def contacts_callback(self, msg):
            # TODO: Update this logic to match your robot's contact sensor message structure
            # Here we assume msg.states is a list of ContactsState for each foot
            with self.lock:
                for i in range(4):
                    if i < len(msg.states):
                        feet_contacts[i] = (len(msg.states[i].contact_positions) > 0)
                        # Use the norm of the first contact normal force if available
                        if len(msg.states[i].contact_normals) > 0:
                            feet_forces[i] = np.linalg.norm(msg.states[i].contact_normals[0])
                        else:
                            feet_forces[i] = 0.0
                    else:
                        feet_contacts[i] = False
                        feet_forces[i] = 0.0

        def joint_callback(self, msg):
            with self.lock:
                joint_angles[:] = np.array(msg.position)
                joint_velocities[:] = np.array(msg.velocity)

        def odom_callback(self, msg):
            with self.lock:
                base_position[:] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
                base_linear_velocity[:] = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
                # Orientation (roll, pitch, yaw) can be extracted from quaternion if needed

    rclpy.init()
    feedback_node = GazeboFeedback()
    feedback_thread = threading.Thread(target=rclpy.spin, args=(feedback_node,), daemon=True)
    feedback_thread.start()

    # Publisher for joint torques (replace topic name/type as needed for your robot)
    pub_node = rclpy.create_node('cpg_command_publisher')
    joint_pub = pub_node.create_publisher(Float64MultiArray, '/joint_command', 10)

    cpg = HopfNetwork(time_step=TIME_STEP, gait=gait_name)
    TEST_STEPS = int(simulation_time / TIME_STEP)
    t = np.arange(TEST_STEPS) * TIME_STEP
    cpg_history = np.zeros((TEST_STEPS, 4, 4))
    action_history = np.zeros((TEST_STEPS, 12))
    history_leg_ind = 0
    history_leg_motor_ids = [i * 4 + history_leg_ind for i in range(3)]
    foot_desire_history = np.zeros((TEST_STEPS, 3))
    foot_real_history = np.zeros((TEST_STEPS, 3))
    angles_desire_history = np.zeros((TEST_STEPS, 3))
    angles_real_history = np.zeros((TEST_STEPS, 3))
    velocity_history = np.zeros((TEST_STEPS, 3))
    history_phase_change = [0]
    last_state = cpg.state[history_leg_ind]
    enery_cost = 0.0
    orientation_history = np.zeros((TEST_STEPS, 3))
    contact_history = np.zeros((TEST_STEPS, 4))
    speed_history = np.zeros((TEST_STEPS, 3))

    kp = np.array([150, 70, 70])
    kd = np.array([2, 0.5, 0.5])
    kpCartesian = np.diag([2500] * 3)
    kdCartesian = np.diag([40] * 3)

    total_starter = time.time()
    for j in range(TEST_STEPS):
        starter = time.time()
        action = np.zeros(12)
        # Get feedback from Gazebo
        with feedback_node.lock:
            roll, pitch, yaw = base_orientation
            q_last = joint_angles.copy()
            dq = joint_velocities.copy()
            feetInContactBool = feet_contacts.copy()
            feetNormalForces = feet_forces.copy()
            linear_vel = base_linear_velocity.copy()
            cur_position = base_position.copy()
        orientation_history[j] = np.array((roll, pitch, yaw))
        # If you have contact info, update CPG with it
        if USE_FEEDBACK:
            xs, zs = cpg.update(body_ori=(roll, pitch, yaw), contactInfo=(feetInContactBool, feetNormalForces))
        else:
            xs, zs = cpg.update(body_ori=(roll, pitch, yaw))
        ys = foot_y * sideSign
        if notpureforward or TRACK_DIRECTION:
            forward_direction = gait_direction - yaw
            ys += np.sin(forward_direction) * xs
            xs = np.cos(forward_direction) * xs
        xs += foot_x
        # TODO: Replace with actual inverse kinematics and Jacobian from Gazebo or your robot model
        # Use joint angles and velocities from Gazebo feedback
        q = joint_angles.reshape(4, 3)
        dq_leg = joint_velocities.reshape(4, 3)
        for i in range(4):
            tau = np.zeros(3)
            leg_xyz = np.array([xs[i], ys[i], zs[i]])
            # Compute joint angles using IK (replace stub with real IK)
            leg_q = compute_leg_ik(leg_xyz)
            # Compute Jacobian for this leg
            J = compute_leg_jacobian(leg_q)
            # Compute actual foot position and velocity using FK
            foot_pos_real = compute_foot_fk(q[i])
            foot_vel_real = compute_foot_velocity(q[i], dq_leg[i])
            if ADD_JOINT_PD:
                tau += kp * (leg_q - q[i]) + kd * -dq_leg[i]
            if ADD_CARTESIAN_PD:
                pos = foot_pos_real
                vel = foot_vel_real
                tau += np.matmul(J.T, np.matmul(kpCartesian, (leg_xyz - pos)) + np.matmul(kdCartesian, -vel))
            action[3 * i:3 * i + 3] = tau
            if i == history_leg_ind:
                foot_desire_history[j, :] = leg_xyz
                angles_desire_history[j, :] = leg_q
                foot_real_history[j, :] = foot_pos_real
                angles_real_history[j, :] = leg_q
        # Publish joint torques to Gazebo
        msg = Float64MultiArray()
        msg.data = action.tolist()
        joint_pub.publish(msg)
        # Save CPG and robot states
        cpg_history[j, :] = np.concatenate((cpg.X, cpg.X_dot), axis=0)
        contact_history[j, :] = 2.0 * np.array(feetInContactBool)
        action_history[j, :] = action
        velocity_history[j, :] = linear_vel
        cur_state = cpg.state[history_leg_ind]
        if cur_state != last_state:
            last_state = cur_state
            history_phase_change.append(j)
        enery_cost += compute_energy_cost(action, dq)
        speed_history[j, :] = linear_vel
        loop_time = time.time() - total_starter
        if loop_time < TIME_STEP / PLAY_SPEED * (j + 1):
            time.sleep(TIME_STEP / PLAY_SPEED * (j + 1) - loop_time)
    distance_traveled = np.linalg.norm(cur_position)
    print("Energy Cost:", enery_cost)
    print("Distance Traveled:", distance_traveled)
    print("Cost of Transport:", enery_cost / (distance_traveled * GRAVITY * MASS))
    print("average abs(pitch):", np.mean(abs(orientation_history[:, 1])))
    print("orientation variance:", np.var(orientation_history, axis=0))
    print("average x-y speed(last50%)", np.mean(np.linalg.norm(speed_history[int(len(t) / 2):, :2], axis=1)))

    import matplotlib.pyplot as plt
    leg_names = ["FR", "FL", "RR", "RL"]
    scales = [1, np.pi, 10, 100]
    fig = plt.figure()
    for i in range(4):
        plt.subplot(4, 1, 1 + i)
        plt.title(leg_names[i])
        for j in range(4):
            plt.plot(t, cpg_history[:, j, i] / scales[j])
        plt.legend(["r", "theta", "dr", "dtheta"])
    plt.tight_layout()

    fig = plt.figure()
    plt.title("Position " + leg_names[history_leg_ind] + " Real vs Desire")
    plt.plot(t, foot_real_history)
    plt.plot(t, foot_desire_history)
    plt.legend(["real_x", "real_y", "real_z", "desire_x", "desire_y", "desire_z"])

    fig = plt.figure()
    plt.title("Angles " + leg_names[history_leg_ind] + " Real vs Desire")
    plt.plot(t, angles_real_history)
    plt.plot(t, angles_desire_history)
    plt.legend(["real_thigh", "real_calf", "real_foot", "desire_thigh", "desire_calf", "desire_foot"])

    history_phase_change = np.array(history_phase_change)
    history_duration = history_phase_change[1:] - history_phase_change[:-1]
    try:
        start_of_a_cycle = history_phase_change[:-1].reshape(-1, 2)
    except:
        start_of_a_cycle = history_phase_change[:-2].reshape(-1, 2)
    start_of_a_cycle = start_of_a_cycle[:, 0] * TIME_STEP
    k = int(len(history_duration) / 2)
    history_duration = np.array(history_duration[:2 * k]).reshape(-1, 2)
    duty_factors = history_duration[:, 1] / np.sum(history_duration, axis=1)
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Duty Factor")
    plt.plot(start_of_a_cycle, duty_factors)
    plt.subplot(2, 1, 2)
    plt.title("Phase Durations")
    plt.plot(start_of_a_cycle, history_duration[:, 0] * TIME_STEP)
    plt.plot(start_of_a_cycle, history_duration[:, 1] * TIME_STEP)
    plt.legend(["Swing Phase Duration", "Stance Phase Duration"])

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, speed_history)
    plt.grid()
    plt.title("linear speed")
    plt.legend(["x", "y", "z"])
    plt.subplot(2, 1, 2)
    plt.title("x-y speed")
    plt.plot(t, np.linalg.norm(speed_history[:, :2], axis=1))
    plt.grid()

    print("duty factor", duty_factors[-1])
    print("swing duration", history_duration[-1, 0] * TIME_STEP)
    print("stance duration", history_duration[-1, 1] * TIME_STEP)
    plt.show()
