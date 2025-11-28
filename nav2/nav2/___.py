#!/usr/bin/env python3
"""
ROS2 node: waypoint_servo_node (improved)

Publishes incremental Twist commands to /delta_twist_cmds to move an
end-effector to a set of waypoints (position + quaternion).

Features:
 - Servoing loop at CONTROL_HZ
 - Stops (holds) at each real waypoint for HOLD_SECONDS
 - Position & orientation tolerances
 - Optional interpolation between waypoints (no hold on interpolated)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import numpy as np
import math

# ----------------------------- Helper quaternion funcs -----------------------------

def quat_normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_conjugate(q):
    q = np.array(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def small_angle_from_quat_error(target_q, current_q):
    """
    Return small-angle 3-vector approximating orientation error
    between current_q and target_q.
    """
    tq = quat_normalize(target_q)
    cq = quat_normalize(current_q)
    qc = quat_conjugate(cq)
    qe = quat_mul(tq, qc)
    # ensure w positive for shortest rotation
    if qe[3] < 0.0:
        qe = -qe
    # small-angle approximation -> error vector = 2 * (x,y,z)
    return 2.0 * qe[:3]

# ----------------------------- Node implementation -----------------------------

class WaypointServoNode(Node):
    def __init__(self):
        super().__init__('waypoint_servo_node')

        # Pose source topic
        self.CURRENT_POSE_TOPIC = '/tcp_pose_raw'

        # Publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # Subscriber for current EE pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.CURRENT_POSE_TOPIC,
            self.pose_callback,
            10,
        )

        # Control loop settings
        self.CONTROL_HZ = 50.0
        self.control_period = 1.0 / self.CONTROL_HZ
        self.linear_gain = 1.0
        self.angular_gain = 1.0

        # Max speeds (safety)
        self.MAX_LINEAR_SPEED = 0.5   # m/s (norm of linear vector)
        self.MAX_ANGULAR_SPEED = 1.0  # rad/s (norm of angular vector)

        # Tolerances
        self.POS_TOL = 0.15     # meters
        self.ORI_TOL = 0.15     # norm of small-angle vector (rad)

        # Hold at real waypoints
        self.HOLD_SECONDS = 1.0

        # Interpolation for smoothness
        self.INTERP_STEPS = 20

        # Define waypoints (pos [m], quat [x,y,z,w]) — normalized on load
        self.waypoints = [
            ([-0.214, -0.532, 0.557], [0.707, 0.028, 0.034, 0.707]),  # P1
            ([-0.159,  0.501, 0.415], [0.029, 0.997, 0.045, 0.033]),  # P2
            ([-0.806,  0.010, 0.182], [-0.684, 0.726, 0.05, 0.008])   # P3
        ]
        # normalize waypoint quaternions
        self.waypoints = [
            (np.array(p, dtype=float), quat_normalize(np.array(q, dtype=float)))
            for (p, q) in self.waypoints
        ]

        # State
        self.current_pose = None  # tuple (pos numpy3, quat numpy4)
        self._active = True

        # Build plan (interpolated points)
        self.plan = self.build_plan(self.waypoints, steps=self.INTERP_STEPS)
        self.plan_index = 0
        self.holding_until = None  # epoch seconds when hold ends

        # Timer
        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info('WaypointServoNode initialized. Waiting for pose...')

    # ------------------------- Pose callback -------------------------
    def pose_callback(self, msg: PoseStamped):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ], dtype=float)
        self.current_pose = (pos, quat)

    # ------------------------- Planning -------------------------
    def build_plan(self, waypoints, steps=0):
        """
        Build plan as list of (pos[numpy3], quat[numpy4], hold_flag).
        hold_flag=True only for real waypoints
        hold_flag=False for interpolated points.
        """
        plan = []
        for i in range(len(waypoints)):
            p_from, q_from = waypoints[i]
            plan.append((p_from.copy(), q_from.copy(), True))
            if steps > 0 and i < len(waypoints) - 1:
                p_to, q_to = waypoints[i + 1]
                for s in range(1, steps + 1):
                    t = float(s) / float(steps + 1)
                    p_interp = (1.0 - t) * p_from + t * p_to
                    # simple quaternion linear interp + renorm (not slerp) — OK for small steps
                    q_interp = quat_normalize((1.0 - t) * q_from + t * q_to)
                    plan.append((p_interp, q_interp, False))
        return plan

    # ------------------------- Control loop -------------------------
    def control_loop(self):
        if not self._active:
            return

        # ensure we have a current pose
        if self.current_pose is None:
            # waiting for pose input
            return

        # all done?
        if self.plan_index >= len(self.plan):
            self.publish_zero_twist()
            self.get_logger().info('All waypoints done. Stopping.')
            self._active = False
            return

        # handle hold timer
        if self.holding_until is not None:
            now = self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1] / 1e9
            if now >= self.holding_until:
                self.holding_until = None
                self.plan_index += 1
                self.get_logger().info('Hold complete. Next target.')
            else:
                self.publish_zero_twist()
            return

        # current target
        target_pos, target_quat, hold_flag = self.plan[self.plan_index]
        cur_pos, cur_quat = self.current_pose

        # position error vector and norm
        pos_err_vec = target_pos - cur_pos
        pos_err = np.linalg.norm(pos_err_vec)

        # orientation error (small-angle vector) and norm
        ori_err_vec = small_angle_from_quat_error(target_quat, cur_quat)
        ori_err = np.linalg.norm(ori_err_vec)

        # check tolerances
        if pos_err <= self.POS_TOL and ori_err <= self.ORI_TOL:
            if hold_flag:
                self.get_logger().info(
                    f'Reached waypoint {self.plan_index} (pos_err={pos_err:.3f}, ori_err={ori_err:.3f}). Holding...'
                )
                self.publish_zero_twist()
                now = self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1] / 1e9
                self.holding_until = now + self.HOLD_SECONDS
            else:
                # skip interp point
                self.plan_index += 1
            return

        # P controller in Cartesian space (vector control)
        linear_cmd = self.linear_gain * pos_err_vec
        lin_norm = np.linalg.norm(linear_cmd)
        if lin_norm > self.MAX_LINEAR_SPEED and lin_norm > 0.0:
            linear_cmd = linear_cmd * (self.MAX_LINEAR_SPEED / lin_norm)

        angular_cmd = self.angular_gain * ori_err_vec
        ang_norm = np.linalg.norm(angular_cmd)
        if ang_norm > self.MAX_ANGULAR_SPEED and ang_norm > 0.0:
            angular_cmd = angular_cmd * (self.MAX_ANGULAR_SPEED / ang_norm)

        # publish Twist
        self.publish_twist(linear_cmd, angular_cmd)

    # ------------------------- Publishers -------------------------
    def publish_twist(self, linear, angular):
        msg = Twist()
        # ensure length 3 vectors
        lx, ly, lz = float(linear[0]), float(linear[1]), float(linear[2])
        ax, ay, az = float(angular[0]), float(angular[1]), float(angular[2])
        msg.linear.x = lx
        msg.linear.y = ly
        msg.linear.z = lz
        msg.angular.x = ax
        msg.angular.y = ay
        msg.angular.z = az
        self.cmd_pub.publish(msg)

    def publish_zero_twist(self):
        self.cmd_pub.publish(Twist())

# ----------------------------- Main --------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = WaypointServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        node.publish_zero_twist()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
