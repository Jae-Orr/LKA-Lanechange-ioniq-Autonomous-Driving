#! /usr/bin/env python3
import rospy
from std_msgs.msg import Float32, Bool, String
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker
import numpy as np
from math import radians, degrees, atan2, cos, sin, sqrt
from numpy.linalg import norm
# from gps_common.msg import GPSFix
from nav_msgs.msg import Odometry, Path
import tf
import math
from geopy.distance import geodesic
from geometry_msgs.msg import Vector3
from novatel_oem7_msgs.msg import BESTGNSSPOS
from collections import deque
from vehicle_control.msg import Actuator
from visualization_msgs.msg import MarkerArray
from copy import deepcopy
from lidar_tracking.msg import AdjacentVehicle
from std_msgs.msg import String


#for decision
# from AdjacentVehicle.msg import AdjacentVehicle  
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion
from std_msgs.msg import Header




# initialize
V_MAX = 1000000000000000     # maximum velocity [m/s]
ACC_MAX = 100000000000000 # maximum acceleration [m/ss]
K_MAX = 99999999999999999999999999999999999999
TARGET_SPEED = 5.0 # target speed [m/s]
LANE_WIDTH = 5.0 # lane width [m]
COL_CHECK = 0.25 # collision check distance [m]


MIN_T = 5.0 # minimum terminal time [s]
MAX_T = 5.0 # maximum terminal time [s]
DT_T = 0.25 # dt for terminal time [s] : MIN_T 에서 MAX_T 로 어떤 dt 로 늘려갈지를 나타냄
DT = 0.05 # timestep for update

# cost weights
K_J = 0.1 # weight for jerk
K_T = 0.1 # weight for terminal time
K_D = 1.0 # weight for consistency
K_V = 1.0 # weight for getting to target speed
K_LAT = 1.0 # weight for lateral direction
K_LON = 1.0 # weight for longitudinal direction

TARGET_SPEED_DEFAULT = 10
LOOKAHEAD_OFFSET =20


DF_SET_LEFT = np.array([6.6/2])
DF_SET_RIGHT = np.array([-0.7/2])

#############################################################################
# 유틸리티 함수: 좌표 변환 및 Frenet 관련
#############################################################################








def next_waypoint(x, y, mapx, mapy):
    closest_wp = get_closest_waypoints(x, y, mapx, mapy)

    candidate = closest_wp+1
    if candidate >= len(mapx) :
        candidate=closest_wp

    map_vec = [mapx[candidate] - mapx[closest_wp], mapy[candidate] - mapy[closest_wp]]
    ego_vec = [x - mapx[candidate], y - mapy[closest_wp]]

    direction  = np.sign(np.dot(map_vec, ego_vec))

    if direction >= 0:
        next_wp = closest_wp + 1
    else:
        next_wp = closest_wp

    return next_wp

def calc_maps(mapx, mapy):
    maps = [0.0]
    for i in range(1, len(mapx)):
        dx = mapx[i] - mapx[i-1]
        dy = mapy[i] - mapy[i-1]
        maps.append(maps[-1] + np.sqrt(dx**2 + dy**2))
    return np.array(maps)

def get_closest_waypoints(x, y, mapx, mapy):
    min_len = 1e10
    closest_wp = 0

    for i in range(len(mapx)):
        _mapx = mapx[i]
        _mapy = mapy[i]
        dist = get_dist(x, y, _mapx, _mapy)

        if dist < min_len:
            min_len = dist
            closest_wp = i

    return closest_wp

def get_dist(x, y, _x, _y):
    return np.sqrt((x - _x)**2 + (y - _y)**2)


def get_frenet(x, y, mapx, mapy):
    next_wp = next_waypoint(x, y, mapx, mapy)
    prev_wp = next_wp -1

    n_x = mapx[next_wp] - mapx[prev_wp]
    n_y = mapy[next_wp] - mapy[prev_wp]
    x_x = x - mapx[prev_wp]
    x_y = y - mapy[prev_wp]

    # (n_x, n_y) 길이에 대한 정사영된 길이의 비율을 나타낸다. 
    proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y)
    proj_x = proj_norm*n_x # 정사영된 점의 상대 x좌표
    proj_y = proj_norm*n_y # 정사영된 점의 상대 y좌표

    #-------- get frenet d
    frenet_d = get_dist(x_x,x_y,proj_x,proj_y)

    ego_vec = [x-mapx[prev_wp], y-mapy[prev_wp], 0]
    map_vec = [n_x, n_y, 0]
    d_cross = np.cross(ego_vec,map_vec)
    if d_cross[-1] > 0:
        frenet_d = -frenet_d

    #-------- get frenet s
    frenet_s = 0
    for i in range(prev_wp):
        frenet_s = frenet_s + get_dist(mapx[i],mapy[i],mapx[i+1],mapy[i+1])

    frenet_s = frenet_s + get_dist(0,0,proj_x,proj_y)

    return frenet_s, frenet_d


def get_cartesian(s, d, mapx, mapy, maps):
    #maps는 누적거리라고 보면 된다. 
    prev_wp = 0 #일단 초기값을 0으로 둔것임 - 이후 검사를 통해 1씩 늘려 나간다

    # s값이 전체 도로 길이를 초과하는 경우를 대비하여 s를 도로 길이 내로 조정하는 역할
    s = np.mod(s, maps[-2])

    # s가 속하는 구간에서 prev_wp와 다음 인덱스 next_wp를 정하는 작업
    while(s > maps[prev_wp+1]) and (prev_wp < len(maps)-2):
        prev_wp = prev_wp + 1

    # 도로가 순환하는 경우에 이를 처음 인덱스로 다시 조정하기 위함
    next_wp = np.mod(prev_wp+1,len(mapx))

    dx = (mapx[next_wp]-mapx[prev_wp])
    dy = (mapy[next_wp]-mapy[prev_wp])

    heading = np.arctan2(dy, dx) # [rad]

    # the x,y,s along the segment
    seg_s = s - maps[prev_wp]

    seg_x = mapx[prev_wp] + seg_s*np.cos(heading)
    seg_y = mapy[prev_wp] + seg_s*np.sin(heading)

    perp_heading = heading + 90 * np.pi/180
    x = seg_x + d*np.cos(perp_heading)
    y = seg_y + d*np.sin(perp_heading)

    return x, y, heading



# def calc_maps(mapx, mapy):
#     maps = [0.0]
#     for i in range(1, len(mapx)):
#         dx = mapx[i] - mapx[i-1]
#         dy = mapy[i] - mapy[i-1]
#         maps.append(maps[-1] + np.sqrt(dx**2 + dy**2))
#     return np.array(maps)



class QuinticPolynomial:

    def __init__(self, xi, vi, ai, xf, vf, af, T):
        # calculate coefficient of quintic polynomial
        # used for lateral trajectory

        # x(t) = x_0 + (v_0)t + (1/2)(a_0)t^2  
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5*ai

        A = np.array([[T**3, T**4, T**5],
                      [3*T**2, 4*T**3, 5*T** 4],
                      [6*T, 12*T**2, 20*T**3]])
        b = np.array([xf - self.a0 - self.a1*T - self.a2*T**2,
                      vf - self.a1 - 2*self.a2*T,
                      af - 2*self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    # calculate postition info.
    def calc_pos(self, t):
        x = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5 * t ** 5
        return x

    # calculate velocity info.
    def calc_vel(self, t):
        v = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4
        return v

    # calculate acceleration info.
    def calc_acc(self, t):
        a = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3
        return a

    # calculate jerk info.
    def calc_jerk(self, t):
        j = 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2
        return j




class QuarticPolynomial:

    def __init__(self, xi, vi, ai, vf, af, T):
        # calculate coefficient of quartic polynomial
        # used for longitudinal trajectory
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5*ai

        A = np.array([[3*T**2, 4*T**3],
                             [6*T, 12*T**2]])
        b = np.array([vf - self.a1 - 2*self.a2*T,
                             af - 2*self.a2])

        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    # calculate postition info.
    def calc_pos(self, t):
        x = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4
        return x

    # calculate velocity info.
    def calc_vel(self, t):
        v = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3
        return v

    # calculate acceleration info.
    def calc_acc(self, t):
        a = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2
        return a

    # calculate jerk info.
    def calc_jerk(self, t):
        j = 6*self.a3 + 24*self.a4*t
        return j



class FrenetPath:

    def __init__(self):
        # time
        self.t = []

        # lateral traj in Frenet frame
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []

        # longitudinal traj in Frenet frame
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []

        # cost
        self.c_lat = 0.0
        self.c_lon = 0.0
        self.c_tot = 0.0

        # combined traj in global frame
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.kappa = []

def calc_frenet_paths_left(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, opt_d):
    frenet_paths = []

    # generate path to each offset goal
    for df in DF_SET_LEFT:

        # Lateral motion planning
        for T in np.arange(MIN_T, MAX_T+DT_T, DT_T):
            fp = FrenetPath()
            lat_traj = QuinticPolynomial(di, di_d, di_dd, df, df_d, df_dd, T)

            fp.t = [t for t in np.arange(0.0, T, DT)] #0부터 T까지 DT간격으로 시간 값을 리스트에 저장한다.
            fp.d = [lat_traj.calc_pos(t) for t in fp.t]
            fp.d_d = [lat_traj.calc_vel(t) for t in fp.t]
            fp.d_dd = [lat_traj.calc_acc(t) for t in fp.t]
            fp.d_ddd = [lat_traj.calc_jerk(t) for t in fp.t]

            # Longitudinal motion planning (velocity keeping)
            tfp = deepcopy(fp)
            lon_traj = QuarticPolynomial(si, si_d, si_dd, sf_d, sf_dd, T)

            tfp.s = [lon_traj.calc_pos(t) for t in fp.t]
            tfp.s_d = [lon_traj.calc_vel(t) for t in fp.t]
            tfp.s_dd = [lon_traj.calc_acc(t) for t in fp.t]
            tfp.s_ddd = [lon_traj.calc_jerk(t) for t in fp.t]

            # 경로 늘려주기 (In case T < MAX_T)
            for _t in np.arange(T, MAX_T+3*DT, DT):
                tfp.t.append(_t)
                tfp.d.append(tfp.d[-1]) # lateral 위치는 그대로 유지 -> 차선 변경 완료 이후를 의미
                _s = tfp.s[-1] + tfp.s_d[-1] * DT # 마지막 속도에 dt를 곱해서 앞으로 이동한 위치를 의미
                tfp.s.append(_s)

                # lateral 방향은 변함이 없다. 
                tfp.s_d.append(tfp.s_d[-1])
                tfp.s_dd.append(tfp.s_dd[-1])
                tfp.s_ddd.append(tfp.s_ddd[-1])

                tfp.d_d.append(tfp.d_d[-1])
                tfp.d_dd.append(tfp.d_dd[-1])
                tfp.d_ddd.append(tfp.d_ddd[-1])

            J_lat = sum(np.power(tfp.d_ddd, 2))  # lateral jerk
            J_lon = sum(np.power(tfp.s_ddd, 2))  # longitudinal jerk

            # cost for consistency
            d_diff = (tfp.d[-1] - opt_d) ** 2
            # cost for target speed
            v_diff = (TARGET_SPEED - tfp.s_d[-1]) ** 2

            # lateral cost
            tfp.c_lat = K_J * J_lat + K_T * T + K_D * d_diff
            # logitudinal cost
            tfp.c_lon = K_J * J_lon + K_T * T + K_V * v_diff

            # total cost combined
            tfp.c_tot = K_LAT * tfp.c_lat + K_LON * tfp.c_lon

            frenet_paths.append(tfp)

    return frenet_paths

def calc_frenet_paths_right(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, opt_d):
    frenet_paths = []

    # generate path to each offset goal
    for df in DF_SET_RIGHT:

        # Lateral motion planning
        for T in np.arange(MIN_T, MAX_T+DT_T, DT_T):
            fp = FrenetPath()
            lat_traj = QuinticPolynomial(di, di_d, di_dd, df, df_d, df_dd, T)

            fp.t = [t for t in np.arange(0.0, T, DT)] #0부터 T까지 DT간격으로 시간 값을 리스트에 저장한다.
            fp.d = [lat_traj.calc_pos(t) for t in fp.t]
            fp.d_d = [lat_traj.calc_vel(t) for t in fp.t]
            fp.d_dd = [lat_traj.calc_acc(t) for t in fp.t]
            fp.d_ddd = [lat_traj.calc_jerk(t) for t in fp.t]

            # Longitudinal motion planning (velocity keeping)
            tfp = deepcopy(fp)
            lon_traj = QuarticPolynomial(si, si_d, si_dd, sf_d, sf_dd, T)

            tfp.s = [lon_traj.calc_pos(t) for t in fp.t]
            tfp.s_d = [lon_traj.calc_vel(t) for t in fp.t]
            tfp.s_dd = [lon_traj.calc_acc(t) for t in fp.t]
            tfp.s_ddd = [lon_traj.calc_jerk(t) for t in fp.t]

            # 경로 늘려주기 (In case T < MAX_T)
            for _t in np.arange(T, MAX_T+3*DT, DT):
                tfp.t.append(_t)
                tfp.d.append(tfp.d[-1]) # lateral 위치는 그대로 유지 -> 차선 변경 완료 이후를 의미
                _s = tfp.s[-1] + tfp.s_d[-1] * DT # 마지막 속도에 dt를 곱해서 앞으로 이동한 위치를 의미
                tfp.s.append(_s)

                # lateral 방향은 변함이 없다. 
                tfp.s_d.append(tfp.s_d[-1])
                tfp.s_dd.append(tfp.s_dd[-1])
                tfp.s_ddd.append(tfp.s_ddd[-1])

                tfp.d_d.append(tfp.d_d[-1])
                tfp.d_dd.append(tfp.d_dd[-1])
                tfp.d_ddd.append(tfp.d_ddd[-1])

            J_lat = sum(np.power(tfp.d_ddd, 2))  # lateral jerk
            J_lon = sum(np.power(tfp.s_ddd, 2))  # longitudinal jerk

            # cost for consistency
            d_diff = (tfp.d[-1] - opt_d) ** 2
            # cost for target speed
            v_diff = (TARGET_SPEED - tfp.s_d[-1]) ** 2

            # lateral cost
            tfp.c_lat = K_J * J_lat + K_T * T + K_D * d_diff
            # logitudinal cost
            tfp.c_lon = K_J * J_lon + K_T * T + K_V * v_diff

            # total cost combined
            tfp.c_tot = K_LAT * tfp.c_lat + K_LON * tfp.c_lon

            frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fplist, mapx, mapy, maps):

    # transform trajectory from Frenet to Global
    for fp in fplist:
        for i in range(len(fp.s)):
            _s = fp.s[i]
            _d = fp.d[i]
            _x, _y, _ = get_cartesian(_s, _d, mapx, mapy, maps)
            fp.x.append(_x)
            fp.y.append(_y)

        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(np.arctan2(dy, dx))
            fp.ds.append(np.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            yaw_diff = fp.yaw[i + 1] - fp.yaw[i]
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            fp.kappa.append(yaw_diff / fp.ds[i])

    return fplist




def collision_check(fp, obs, mapx, mapy, maps):
    for i in range(len(obs[:, 0])):
        # get obstacle's position (x,y)
        obs_xy = get_cartesian( obs[i, 0], obs[i, 1], mapx, mapy, maps)

        d = [((_x - obs_xy[0]) ** 2 + (_y - obs_xy[1]) ** 2)
             for (_x, _y) in zip(fp.x, fp.y)]

        collision = any([di <= COL_CHECK ** 2 for di in d])

        if collision:
            return True

    return False




def check_path(fplist, obs, mapx, mapy, maps):
    ok_ind = []
    for i, _path in enumerate(fplist):
        acc_squared = [(abs(a_s**2 + a_d**2)) for (a_s, a_d) in zip(_path.s_dd, _path.d_dd)]
        if any([v > V_MAX for v in _path.s_d]):  # Max speed check
            continue
        elif any([acc > ACC_MAX**2 for acc in acc_squared]): # 가속도가 너무 크면 탈락
            continue
        elif any([abs(kappa) > K_MAX for kappa in fplist[i].kappa]):  # Max curvature check
            continue
        elif collision_check(_path, obs, mapx, mapy, maps): # 장애물 충돌 위험시 탈락
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]



def frenet_optimal_planning_left(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, mapx, mapy, maps, opt_d):
    fplist = calc_frenet_paths_left(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, opt_d)
    fplist = calc_global_paths(fplist, mapx, mapy, maps)

    fplist = check_path(fplist, obs, mapx, mapy, maps)
    # find minimum cost path
    min_cost = float("inf")
    opt_traj = None
    opt_ind = 0
    for fp in fplist:
        if min_cost >= fp.c_tot:
            min_cost = fp.c_tot
            opt_traj = fp
            _opt_ind = opt_ind
        opt_ind += 1

    try:
        _opt_ind
    except NameError:
        print(" No solution ! ")

    return fplist, _opt_ind

def frenet_optimal_planning_right(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, mapx, mapy, maps, opt_d):
    fplist = calc_frenet_paths_right(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, opt_d)
    fplist = calc_global_paths(fplist, mapx, mapy, maps)

    fplist = check_path(fplist, obs, mapx, mapy, maps)
    # find minimum cost path
    min_cost = float("inf")
    opt_traj = None
    opt_ind = 0
    for fp in fplist:
        if min_cost >= fp.c_tot:
            min_cost = fp.c_tot
            opt_traj = fp
            _opt_ind = opt_ind
        opt_ind += 1

    try:
        _opt_ind
    except NameError:
        print(" No solution ! ")

    return fplist, _opt_ind

class PurePursuit:
    def __init__(self):
        self.L = 3
        self.k = 0.14  # 0.1~1
        self.Lfc = 6.0
        self.alpha = 1.5
    def euc_distance(self, pt1, pt2):
        return norm([pt2[0] - pt1[0], pt2[1] - pt1[1]])

    def global_to_local(self, target_point, position, yaw):
        dx = target_point[0] - position[0]
        dy = target_point[1] - position[1]

        x_local = dx * cos(-yaw) - dy * sin(-yaw)
        y_local = dx * sin(-yaw) + dy * cos(-yaw) 

        return x_local, y_local 

    def vel_global_to_local(self, target_point, position, yaw):
        dx = target_point[0] - position[0]
        dy = target_point[1] - position[1]

        x_local = dx * cos(-yaw) - dy * sin(-yaw)
        y_local = dx * sin(-yaw) + dy * cos(-yaw)

        return x_local, y_local 

    def run(self, vEgo, target_point, position, yaw, sEgo):
        # gamma = math.tan(radians(abs(sEgo)))/self.L
        # lfc = self.Lfc / (1+self.alpha*gamma)
        lfd = self.Lfc + self.k * vEgo
        lfd = np.clip(lfd, 5, 10)
        rospy.loginfo(f"Lfd: {lfd}")
        x_local , y_local = self.vel_global_to_local(target_point,position, yaw)
        diff = np.sqrt(x_local**2 + y_local**2)
        
        if diff > 0:
            dis = np.linalg.norm(diff)
            if dis >= lfd:
                theta = atan2(y_local, x_local)
                steering_angle = atan2(2 * self.L * sin(theta), lfd)
                return degrees(steering_angle), target_point
        return 0.0, target_point  

    def run_global(self, vEgo, target_point, position, yaw, sEgo):
        # gamma = math.tan(radians(abs(sEgo)))/self.L
       # lfc = self.Lfc/(1+self.alpha*gamma)
        lfd = self.Lfc + self.k * vEgo
        lfd = np.clip(lfd, 10, 11)
        rospy.loginfo(f"Lfd: {lfd}")
        x_local , y_local = self.global_to_local(target_point,position, yaw)
        diff = np.sqrt(x_local**2 + y_local**2)
        rospy.loginfo(f"diff: {diff}")
        theta = atan2(y_local, x_local)
        steering_angle = atan2(2 * self.L * sin(theta), lfd)
        return degrees(steering_angle), target_point 

class PID:
    def __init__(self, kp, ki, kd, dt=0.05):
        self.K_P = kp
        self.K_I = ki
        self.K_D = kd
        self.pre_error = 0.0
        self.integral_error = 0.0
        self.dt = dt

    def run(self, target, current):
        error = sqrt((target[0] - current[0])**2 + (target[1] - current[1])**2)
        derivative_error = (error - self.pre_error) / self.dt
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -5, 5)

        pid = self.K_P * error + self.K_I * self.integral_error + self.K_D * derivative_error
        pid = np.clip(pid, -100, 100)
        self.pre_error = error
        return pid

class Start:
    def __init__(self):
        self.pure_pursuit = PurePursuit()
        self.pid = PID(kp=1.0, ki=0.1, kd=0.01)
        self.local_path_sub = rospy.Subscriber('/ego_waypoint', Marker, self.local_cb)
        self.global_gps_sub = rospy.Subscriber('/current_global_waypoint', Marker, self.global_cb)
        self.pose_sub = rospy.Subscriber('/ego_pos', PoseStamped, self.pose_cb)
        # self.vel_sub = rospy.Subscriber('/vehicle/curr_v', Float32, self.vel_cb)
        self.gps_pos_sub = rospy.Subscriber('/gps_ego_pose',Point,self.gps_pos_cb)
        # self.odom_sub = rospy.Subscriber('/novatel/oem7/odom', Odometry, self.odom_cb,queue_size=20)
        self.start_flag_sub = rospy.Subscriber('/start_flag',Bool, self.flag_cb)
        self.point_sub = rospy.Subscriber('/last_target_point', Marker, self.point_callback)
        self.gps_sub = rospy.Subscriber('/novatel/oem7/bestgnsspos', BESTGNSSPOS, self.bestgps_cb)
        self.yaw_sub = rospy.Subscriber('/vehicle/yaw_rate_sensor',Float32,self.yaw_cb)
        self.rl_sub = rospy.Subscriber('/vehicle/velocity_RL', Float32, self.rl_callback)
        self.rr_sub = rospy.Subscriber('/vehicle/velocity_RR', Float32, self.rr_callback)
        self.steer_sub = rospy.Subscriber('/vehicle/steering_angle', Float32, self.steer_callback)
        self.actuator_pub = rospy.Publisher('/target_actuator', Actuator, queue_size=10)
        self.light_pub = rospy.Publisher('/vehicle/left_signal', Float32, queue_size =10)
        self.global_odom_pub = rospy.Publisher('/global_odom_frame_point',Marker,queue_size=10)
        self.laps_complete_pub = rospy.Publisher('/laps_completed',Bool,queue_size=10)
        
        self.obstacle_sub = rospy.Subscriber('/mobinha/hazard_warning',Bool,self.obstacle_cb)
        self.sign_sub = rospy.Subscriber('/mobinha/is_crossroad',Bool, self.sign_cb)

        # 차선 변경을 위해 추가한 토픽
        self.lc_sub = rospy.Subscriber('/lane_change_cmd', String,self.lane_change_cmd_callback)
        self.points_sub = rospy.Subscriber('/target_points', MarkerArray,self.points_callback)
        self.lanechange_path_pub = rospy.Publisher('/lanechange_path', Marker, queue_size=10)
        self.ego_marker_pub = rospy.Publisher('/ego_position_marker', Marker, queue_size=10)

        # 판단을 위해 추가한 토픽 #토픽 이름 수정 필요
        self.adj_vehicle_sub = rospy.Subscriber('/lidar_tracking/adjacent_vehicles',AdjacentVehicle,self.adjacent_vehicle_cb)
        self.vehicle_marker_pub = rospy.Publisher("/adjacent_vehicle_lines", Marker, queue_size=10)
        self.lc_status_pub = rospy.Publisher('/lane_change_status', String, queue_size=1)


        self.curr_v = 0
        self.pose = PoseStamped()
        self.global_waypoints_x = None
        self.global_waypoints_y = None
        self.local_waypoint = Point()
        self.steer_ratio = 12
        self.current_point = None
        self.curr_lat = None
        self.curr_lon = None
        self.current_waypoint_idx = 0

        self.global_pose_x = None
        self.global_pose_y = None
        self.yaw_rate = None
        self.is_start = False

        self.moving_average_window = 1  
        self.point_history_x = deque(maxlen=self.moving_average_window)
        self.point_history_y = deque(maxlen=self.moving_average_window)

        self.rl_v = 0
        self.rr_v = 0
        
        self.curr_steer = 0
        self.inter_steer = 0

        self.obstacle_flag = False
        self.sign_flag = False

        #새로 추가한 변수 for lane change
        self.lane_change_flag   = None

        # 1) 카메라에서 받은 “짧은 경로” 저장용
        self.camera_path       = []     # [(x1,y1), (x2,y2), …]
        self.camera_path_x     = []     # [x1, x2, …]
        self.camera_path_y     = []     # [y1, y2, …]
        self.camera_maps       = None   # calc_maps(self.camera_path_x, self.camera_path_y) 결과

        # 2) 차선 변경 궤적 저장 및 완료 플래그
        self.opt_traj           = None  # 한번 생성한 Frenet 경로 저장
        self.lane_change_done   = True  # 기본엔 True, 명령 받으면 False로 전환

        # 3) 목표 lateral 오프셋 (L/R 명령 시 설정할 값)
        self.target_lane_offset = 0.0   # 예: L이면 +0.2, R이면 -0.2

        # 판단 함수를 위해 추가한 변수
        self.region_flags = [False] * 10
        self.vehicle_move_lines = []

        # 인식 기반의 차선 변경 명령을 위한 타이머
        self.auto_cmd_time=None
        self.auto_cmd_duration = 0.5


        self.obs = np.empty((0,2))

    def obstacle_cb(self,msg):
        self.obstacle_flag = msg.data
    
    def sign_cb(self,msg):
        self.sign_flag = msg.data

    def steer_callback(self,msg):
        self.curr_steer = msg.data

    def rl_callback(self,msg):
        self.rl_v = msg.data

    def rr_callback(self,msg):
        self.rr_v = msg.data


    def get_yaw_from_pose(self, pose):
        orientation_q = pose.pose.orientation
        quaternion = (
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        )
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        return yaw

    def bestgps_cb(self,msg):
        self.curr_lat = msg.lat
        self.curr_lon = msg.lon

    def flag_cb(self,msg):
        self.is_start = msg.data

    def yaw_cb(self,msg):
        self.yaw_rate = radians(msg.data)

    def gps_pos_cb(self,msg):
        self.global_pose_x = msg.x
        self.global_pose_y = msg.y

    def point_callback(self,msg):
        self.current_point = Point()
        self.current_point.x = msg.pose.position.x
        self.current_point.y = msg.pose.position.y + 0.2

        # self.point_history_x.append(self.current_point.x)
        # self.point_history_y.append(self.current_point.y)

        # self.current_point.x = sum(self.point_history_x) / len(self.point_history_x)
        # self.current_point.y = sum(self.point_history_y) / len(self.point_history_y)

    # def vel_cb(self, msg):
    #     self.curr_v = msg.data

    def local_cb(self, msg):
        self.local_waypoint = Point()
        x_ = msg.pose.position.x
        y_ = msg.pose.position.y
        self.local_waypoint.x = x_
        self.local_waypoint.y = y_

    def global_cb(self, msg):
        self.global_waypoints_x = msg.pose.position.x
        self.global_waypoints_y = msg.pose.position.y

    def pose_cb(self, msg):
        self.pose = msg
        self.yaw = self.get_yaw_from_pose(self.pose)


    def lane_change_cmd_callback(self, msg):
        if self.auto_cmd_time is not None:
            # 아직 자동 모드 유지 중이므로, 콜백 들어와도 처리하지 않고 리턴
            return

        # 자동 모드가 아닌 상태(=사용자 입력을 받아도 좋음)
        if msg.data in ['L','R']:
            rospy.loginfo(f"[사용자 입력] 명령 수신: {msg.data}")
            self.lane_change_flag = msg.data
            self.lc_status_pub.publish(String(data=f"Lanechange command ({msg.data})"))

        elif msg.data == 'O':
            # 'O' 메시지가 들어오면, 수동 모드에서만 해제
            self.lane_change_flag = None
        else:
            rospy.logwarn(f"잘못된 명령: {msg.data}")

    def points_callback(self, msg: MarkerArray):
        if self.global_pose_x is None or self.global_pose_y is None:
            return 

        # 차량 위치 및 자세 (절대좌표 + yaw)
        ego_x = self.global_pose_x 
        ego_y = self.global_pose_y 
        yaw = self.yaw  # rad

        rel_x_list = []
        rel_y_list = []

        for marker in msg.markers:
            rel_x_list.append(marker.pose.position.x)
            rel_y_list.append(marker.pose.position.y)

        if len(rel_x_list) < 2:
            rospy.logwarn("차선 점이 2개 이상 필요합니다.")
            return

        # 시작점과 끝점으로 방향 벡터 구하기
        dx = rel_x_list[-1] - rel_x_list[0]
        dy = rel_y_list[-1] - rel_y_list[0]
        norm_dir = math.hypot(dx, dy)
        if norm_dir == 0:
            rospy.logwarn("시작점과 끝점이 같습니다.")
            return

        dir_x = dx / norm_dir
        dir_y = dy / norm_dir

        # 끝점부터 일정 간격으로 60개 점 연장
        for i in range(1, 61):
            rel_x_list.append(rel_x_list[-1] + dir_x * 1.1)
            rel_y_list.append(rel_y_list[-1] + dir_y * 1.1)

        self.camera_path_x = []
        self.camera_path_y = []

        for rel_x, rel_y in zip(rel_x_list, rel_y_list):
            abs_x = rel_x * cos(yaw) - rel_y * sin(yaw) + ego_x
            abs_y = rel_x * sin(yaw) + rel_y * cos(yaw) + ego_y
            self.camera_path_x.append(abs_x)
            self.camera_path_y.append(abs_y)        


    #판단 함수 
    def should_delay_lane_change_by_point_check_left(self):
        """
        Trajectory 점 단위로 주변 차량의 경로 선분과 충돌 가능성 판단

        Returns:
            True  → 차선 변경 지연 필요
            False → 차선 변경 가능
        """

        # 1. 위험 구역 검사
        danger_zones = [0, 1, 2]
        for i in danger_zones:
            if self.region_flags[i]:
                rospy.logwarn(f"위험 구역 {i+1}에 차량 존재 → 차선 변경 지연")
                # 거부 시 텍스트 퍼블리시
                self.lc_status_pub.publish(String(data="Lane change rejected"))
                # 1초 뒤 “No lanechange command”로 상태 리셋
                rospy.Timer(rospy.Duration(1.0), self._reset_lanechange_status, oneshot=True)
                return True

        # 2. 경로 유효성 검사
        if self.opt_traj is None:
            rospy.logwarn("판단 불가: opt_traj 없음 → 차선 변경 지연 처리")
            # 거부 시 텍스트 퍼블리시
            self.lc_status_pub.publish(String(data="Lane change rejected"))
            # 1초 뒤 “No lanechange command”로 상태 리셋
            rospy.Timer(rospy.Duration(1.0), self._reset_lanechange_status, oneshot=True)
            return True

        traj_points = list(zip(self.opt_traj.x, self.opt_traj.y))

        # 3. 주변 차량 선분과의 충돌 판단
        # 3. 주변 차량 끝점 x좌표 기반 판단 (상대 좌표 기준)
        for vehicle_line in self.vehicle_move_lines:
            v_end_x = vehicle_line[1][0]  # 끝점의 x좌표 (relative)

            rospy.logwarn(f"차량 끝점 x좌표: {v_end_x:.2f}")
            if v_end_x >= 0.0:
                rospy.logwarn(f"x=0 이상 → 차선 변경 지연")
                # 거부 시 텍스트 퍼블리시
                self.lc_status_pub.publish(String(data="Lane change rejected"))
                # 1초 뒤 “No lanechange command”로 상태 리셋
                rospy.Timer(rospy.Duration(1.0), self._reset_lanechange_status, oneshot=True)
                return True

        # 충돌 없음 → 차선 변경 가능
        return False

    def should_delay_lane_change_by_point_check_right(self):
        """
        Trajectory 점 단위로 주변 차량의 경로 선분과 충돌 가능성 판단

        Returns:
            True  → 차선 변경 지연 필요
            False → 차선 변경 가능
        """

        # 1. 위험 구역 검사
        danger_zones = [4, 5, 6]
        for i in danger_zones:
            if self.region_flags[i]:
                rospy.logwarn(f"위험 구역 {i+1}에 차량 존재 → 차선 변경 지연")
                # 거부 시 텍스트 퍼블리시
                self.lc_status_pub.publish(String(data="Lane change rejected"))
                # 1초 뒤 “No lanechange command”로 상태 리셋
                rospy.Timer(rospy.Duration(1.0), self._reset_lanechange_status, oneshot=True)
                return True

        # 2. 경로 유효성 검사
        if self.opt_traj is None:
            rospy.logwarn("판단 불가: opt_traj 없음 → 차선 변경 지연 처리")
            # 거부 시 텍스트 퍼블리시
            self.lc_status_pub.publish(String(data="Lane change rejected"))
            # 1초 뒤 “No lanechange command”로 상태 리셋
            rospy.Timer(rospy.Duration(1.0), self._reset_lanechange_status, oneshot=True)
            return True

        traj_points = list(zip(self.opt_traj.x, self.opt_traj.y))

   
        # 3. 주변 차량 끝점 x좌표 기반 판단 (상대 좌표 기준)
        for vehicle_line in self.vehicle_move_lines:
            v_end_x = vehicle_line[1][0]  # 끝점의 x좌표 (relative)

            rospy.logwarn(f"차량 끝점 x좌표: {v_end_x:.2f}")
            if v_end_x >= 0.0:
                rospy.logwarn(f"x=0 이상 → 차선 변경 지연")
                # 거부 시 텍스트 퍼블리시
                self.lc_status_pub.publish(String(data="Lane change rejected"))
                # 1초 뒤 “No lanechange command”로 상태 리셋
                rospy.Timer(rospy.Duration(1.0), self._reset_lanechange_status, oneshot=True)
                return True

        # 충돌 없음 → 차선 변경 가능
        return False
    

    def _reset_lanechange_status(self, event):
       # 1초 후 OverlayText를 기본값(“No lanechange command”)으로
       self.lc_status_pub.publish(String(data="No lanechange command"))


    # def points_callback(self, msg: MarkerArray):
    #     if self.global_pose_x is None or self.global_pose_y is None:
    #         return 

    #     # 차량 위치 및 자세 (절대좌표 + yaw)
    #     ego_x = self.global_pose_x 
    #     ego_y = self.global_pose_y 
    #     yaw = self.yaw  # rad

    #     rel_x_list = []
    #     rel_y_list = []

    #     for marker in msg.markers:
    #         rel_x_list.append(marker.pose.position.x)
    #         rel_y_list.append(marker.pose.position.y)
        

    #     last_rel_x = rel_x_list[-1]
    #     last_rel_y = rel_y_list[-1]

    #     for i in range(1,61):
    #         rel_x_list.append(last_rel_x + 1.1 * i)
    #         rel_y_list.append(last_rel_y)

        

    #     self.camera_path_x = []
    #     self.camera_path_y = []

    #     for rel_x, rel_y in zip(rel_x_list, rel_y_list):

    #         # 전역 좌표계로 변환
    #         abs_x = rel_x * cos(yaw) - rel_y * sin(yaw) + ego_x
    #         abs_y = rel_x * sin(yaw) + rel_y * cos(yaw) + ego_y

    #         self.camera_path_x.append(abs_x)
    #         self.camera_path_y.append(abs_y)

        

    def generate_global_waypoints(self):
        if not self.global_waypoints or not self.pose:
            return None

        curr_x = self.pose.pose.position.x
        curr_y = self.pose.pose.position.y

        closest_idx = None
        closest_dist = float('inf')
        for i, waypoint in enumerate(self.global_waypoints):
            dist = np.sqrt((curr_x - waypoint[0])**2 + (curr_y - waypoint[1])**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        if closest_idx is None:
            rospy.logwarn("No closest waypoint found.")
            return None

        global_waypoints = self.global_waypoints[closest_idx]
        return global_waypoints

    

    def global_to_local(self, target_point, position, yaw):
        dx = target_point[0] - position[0]
        dy = target_point[1] - position[1]

        x_local = dx * cos(-yaw) - dy * sin(-yaw)
        y_local = dx * sin(-yaw) + dy * cos(-yaw)

        return x_local, y_local 


    def pub_global_waypoint(self,x,y):
        marker = Marker()
        marker.header.frame_id = "map" 
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = 0 
        marker.type = Marker.SPHERE 
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 5 
        marker.scale.y = 5
        marker.scale.z = 5

        marker.color.a = 1.0  
        marker.color.r = 1.0 
        marker.color.g = 1.0 
        marker.color.b = 1.0

        self.global_odom_pub.publish(marker)

    def pub_lanechange_path(self, traj):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "lanechange_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.3  # 선 굵기
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0  # 빨간색 경로

        for x, y in zip(traj.x, traj.y):
            p = Point() 
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)

        self.lanechange_path_pub.publish(marker)

    def pub_ego_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "ego_vehicle"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0  # 초록색 구체로 표시

        self.ego_marker_pub.publish(marker)


    def adjacent_vehicle_cb(self, msg):
            self.region_flags = msg.region_flags
            poses = msg.ar_PoseVehicles.poses

            ego_x = self.global_pose_x
            ego_y = self.global_pose_y
            yaw = self.yaw  # 라디안

            self.vehicle_move_lines = []  # 새로 수신되었을 때마다 갱신

            for i in range(0, len(poses), 2):
                pose_now = poses[i]
                pose_fut = poses[i + 1]

                # 상대 좌표 그대로 사용
                x0, y0 = pose_now.position.x, pose_now.position.y
                x1, y1 = pose_fut.position.x, pose_fut.position.y

                move_line = [[x0, y0], [x1, y1]]
                self.vehicle_move_lines.append(move_line)
               
            rospy.loginfo(f"[AdjacentVehicle] region_flags = {self.region_flags}")
            rospy.loginfo(f"감지된 차량 수: {len(self.vehicle_move_lines)}")
        


    def run_control_loop(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            

            if self.auto_cmd_time is not None:
                elapsed = rospy.Time.now().to_sec() - self.auto_cmd_time
                if elapsed >= self.auto_cmd_duration:
                    # 0.5초(=auto_cmd_duration) 지났으면 자동으로 'O' 복귀
                    rospy.loginfo("자동차선 변경 명령 유지시간 종료 → 'O'로 복귀")
                    self.lane_change_flag = 'O'
                    self.auto_cmd_time = None

            if not self.is_start:
                rospy.loginfo("Not setting yet...")
                rate.sleep()
                continue

            if self.global_pose_x is not None and self.global_pose_y is not None:
                self.pub_ego_marker(self.global_pose_x, self.global_pose_y)

            if self.region_flags[9] == False and self.region_flags[8] == True:#######################
                # lane_change_flag가 None 또는 'O'(대기 모드)이고,
                # 이전 차선 변경이 완료(self.lane_change_done == True) 상태이면 자동으로 진입
                if (self.lane_change_flag is None or self.lane_change_flag == 'O') and self.lane_change_done:
                    rospy.loginfo("조건 충족: 8번 구역 Empty, 9번 구역 Occupied → 자동 ‘L’ 차선 변경")
                    self.lane_change_flag = 'L'
                    self.auto_cmd_time = rospy.Time.now().to_sec()
                    self.lc_status_pub.publish(String(data="Auto lane change command (L)"))

                    rospy.loginfo("자동 플래그 설정 후 0.1초 대기 → camera_path 콜백 보장")
                    rospy.sleep(0.1)
            
            self.curr_v = (self.rl_v + self.rr_v)/7.2

            if self.lane_change_flag == 'O' and self.lane_change_done :  ####################
                self.lc_status_pub.publish(String(data="No lanechange command"))

            if self.obstacle_flag:
                temp = Actuator()
                temp.accel = 0
                temp.brake = np.clip(100 / max(self.curr_v, 0.1), 0, 100)
                temp.is_waypoint = 0
                self.actuator_pub.publish(temp)
                self.obstacle_flag = False
                rospy.logwarn("Obstacle detect, Command brake")
                continue

    

            if self.lane_change_flag in ['L','R'] and self.lane_change_done:
                rospy.loginfo("Path planing")
                

                self.lane_change_done = False
                self.opt_traj = None

                x, y = self.global_pose_x, self.global_pose_y
                yaw = self.yaw

                
                

                self.camera_maps = calc_maps(self.camera_path_x,self.camera_path_y)
                s, d = get_frenet(x, y, self.camera_path_x, self.camera_path_y)
                # 공통 초기 상태
                si = s
                si_d = self.curr_v
                si_dd = 0.0
                sf_d = TARGET_SPEED
                sf_dd = 0.0
                di = d
                di_d = 0.0
                di_dd = 0.0
                df_d = 0.0
                df_dd = 0.0

                if self.lane_change_flag == 'L':
                    self.target_lane_offset = DF_SET_LEFT[0]
                    self.obs= np.empty((0,2))
                    path_list, opt_ind = frenet_optimal_planning_left(
                        s, 6, 0,                   # si, si_d, si_dd
                        TARGET_SPEED_DEFAULT, 0,   # sf_d, sf_dd
                        d, 0, 0,                   # di, di_d, di_dd
                        0, 0,                      # df_d, df_dd
                        self.obs,                  # obstacle data
                        self.camera_path_x,        # mapx
                        self.camera_path_y,        # mapy
                        self.camera_maps,          # maps
                        self.target_lane_offset    # opt_d
                    )
                    self.opt_traj = path_list[opt_ind]
                    self.pub_lanechange_path(self.opt_traj)
                    rospy.loginfo("Lane change trajectory 생성 완료.")
                    
                    if self.should_delay_lane_change_by_point_check_left():
                        rospy.loginfo("차선 변경 조건 불충족 → 변경 지연")
                        self.opt_traj = None  # 경로 제거
                        self.lane_change_flag = 'O'
                        self.lane_change_done = True  # 상태 업데이트
                        continue
                    else:
                        rospy.loginfo("차선 변경 실행 가능 → 계속 진행")

                    
                    # 이후 차량 제어 로직 계속 진행
                    length = len(self.opt_traj.x)
                    rospy.loginfo(f"length: {length}")
            

                    for i in range(length-1):
                        x1, y1 = self.opt_traj.x[i], self.opt_traj.y[i]
                        x2, y2 = self.opt_traj.x[i+1], self.opt_traj.y[i+1]
                        dist = np.sqrt((x2-x1)**2+(y2-y1)**2)
                        rospy.loginfo(f"[{i} to {i+1}] 거리 : {dist: .3f} m")
                    continue

                elif self.lane_change_flag == 'R':
                    self.target_lane_offset = DF_SET_RIGHT[0]
                    path_list, opt_ind = frenet_optimal_planning_right(
                        s, 6, 0,                   # si, si_d, si_dd
                        TARGET_SPEED_DEFAULT, 0,   # sf_d, sf_dd
                        d, 0, 0,                   # di, di_d, di_dd
                        0, 0,                      # df_d, df_dd
                        self.obs,                  # obstacle data
                        self.camera_path_x,        # mapx
                        self.camera_path_y,        # mapy
                        self.camera_maps,          # maps
                        self.target_lane_offset    # opt_d
                    )
                    self.opt_traj = path_list[opt_ind]
                    self.pub_lanechange_path(self.opt_traj)
                    rospy.loginfo("Lane change trajectory 생성 완료.")

                    if self.should_delay_lane_change_by_point_check_right():
                            rospy.loginfo("차선 변경 조건 불충족 → 변경 지연")
                            rospy.loginfo("차선 변경 조건 불충족 → 변경 지연")
                            rospy.loginfo("차선 변경 조건 불충족 → 변경 지연")
                            rospy.loginfo("차선 변경 조건 불충족 → 변경 지연")
                            self.opt_traj = None  # 경로 제거
                            self.lane_change_flag = 'O'
                            self.lane_change_done = True  # 상태 업데이트
                            continue
                    else:
                        rospy.loginfo("차선 변경 실행 가능 → 계속 진행")
                        rospy.loginfo("차선 변경 실행 가능 → 계속 진행")
                        rospy.loginfo("차선 변경 실행 가능 → 계속 진행")
                        rospy.loginfo("차선 변경 실행 가능 → 계속 진행")

                    length = len(self.opt_traj.x)
                    rospy.loginfo(f"length: {length}")

                    # for i in range(length-1):
                    #     x1, y1 = self.opt_traj.x[i], self.opt_traj.y[i]
                    #     x2, y2 = self.opt_traj.x[i+1], self.opt_traj.y[i+1]
                    #     dist = np.sqrt((x2-x1)**2+(y2-y1)**2)
                    #     rospy.loginfo(f"[{i} to {i+1}] 거리 : {dist: .3f} m")
                    
                    continue

            elif self.lane_change_done == False :
                x, y = self.global_pose_x, self.global_pose_y
                yaw = self.yaw
                position = (x,y)
                if self.opt_traj is None :
                    rospy.logwarn("opt_Traj is None 차선 변경 지연")
                    rate.sleep()
                    continue
                idx = get_closest_waypoints(x, y, self.opt_traj.x, self.opt_traj.y)
                if idx >= len(self.opt_traj.x) - 18:
                    self.lane_change_done = True
                    rospy.loginfo("Lane change 완료, 글로벌 경로로 복귀합니다.")
                    rospy.loginfo("Lane change 완료, 글로벌 경로로 복귀합니다.")
                    rospy.loginfo("Lane change 완료, 글로벌 경로로 복귀합니다.")
                    self.opt_traj = None

                    self.lc_status_pub.publish(String(data="No lanechange command"))
                    rospy.loginfo("Lane change 완료, 상태를 초기화합니다: No lanechange command")

                    continue

                else:
                    if self.opt_traj is None:
                        rospy.logwarn("차선변경 경로 없음")
                        rospy.logwarn("차선변경 경로 없음")
                        rospy.logwarn("차선변경 경로 없음")
                        rospy.logwarn("차선변경 경로 없음")
                        rospy.logwarn("차선변경 경로 없음")
                    rospy.logwarn("차선변경 제어 시작")
                    rospy.logwarn("차선변경 제어 시작")
                    rospy.logwarn("차선변경 제어 시작")
                    
                    tgt_idx = min(idx + LOOKAHEAD_OFFSET, len(self.opt_traj.x) - 18)
                    target_point = (self.opt_traj.x[tgt_idx], self.opt_traj.y[tgt_idx])
                    target_steering, target_position = self.pure_pursuit.run_global(self.curr_v, target_point, position, yaw,self.curr_steer)
                    throttle = self.pid.run(target_position, position)
                    throttle = np.clip(throttle,0,20)
                    accel = throttle
                    b = self.curr_v

                    target_steering = np.clip(target_steering,-35,35)
                    if idx >= 1 and idx <= 18:
                        target_steering = np.clip(target_steering,-3,3)

                        steer_interpolate = np.linspace(self.inter_steer, steer, 10)
                        param = np.clip(self.curr_v,4.6,9)
                        for s in steer_interpolate:
                            temp = Actuator()
                            temp.accel = accel / 1.2
                            temp.steer = s
                            temp.brake = b*2
                            temp.is_waypoint = 1
                            self.actuator_pub.publish(temp)
                            rate.sleep()

                        self.inter_steer = steer   
                        light = Float32()
                        light.data = 0
                        self.light_pub.publish(light) 

                        rospy.loginfo(f"target steering: {target_steering}")
                        
                        self.global_waypoints_x = None
                        self.global_waypoints_y = None
                        continue

                    if idx >= len(self.opt_traj.x)  - 21 and idx <= len(self.opt_traj.x) - 19:
                        target_steering = np.clip(target_steering,-3,3)
                    steer = target_steering * self.steer_ratio/2
                    

                    steer_interpolate = np.linspace(self.inter_steer, steer, 10)


                    param = np.clip(self.curr_v,4,9)
                    for s in steer_interpolate:
                        temp = Actuator()
                        temp.accel = accel / 1.2
                        temp.steer = s
                        temp.brake = 0
                        temp.is_waypoint = 1
                        self.actuator_pub.publish(temp)
                        rate.sleep()
                    
                    
                    
                    self.inter_steer = steer    
                    light = Float32()
                    light.data = 0
                    self.light_pub.publish(light) 

                    rospy.loginfo(f"target steering: {target_steering}")
                    
                    self.global_waypoints_x = None
                    self.global_waypoints_y = None
                    continue

                    

    
            

            # 차선 기반 주행
            elif self.current_point is not None and self.lane_change_done:
                
                way_x = self.current_point.x
                way_y = self.current_point.y

                position = (0, 0)
                waypoint = (way_x, way_y)
                
                yaw = self.yaw
                rospy.loginfo(f"current velocity: {self.curr_v}")
                target_steering, target_position = self.pure_pursuit.run(self.curr_v, waypoint, position, 0 ,self.curr_steer)
                
                throttle = self.pid.run(target_position, position)
                throttle *= 1.5
                throttle = np.clip(throttle,0,20)
                accel = throttle
                steer = target_steering * self.steer_ratio/2
                

                # temp = Actuator()
                # temp.accel = accel / 2.5
                # temp.steer = steer
                # temp.brake = 0
                # temp.is_waypoint = 1

                # rospy.loginfo("Using Local Waypoint")
                
                # self.actuator_pub.publish(temp)
                    
                # self.inter_steer = steer    
                # light = Float32()
                # light.data = 0
                # self.light_pub.publish(light)4
                
                if abs(self.inter_steer-steer)>20:
                    steer_interpolate = np.linspace(self.inter_steer, steer, 6)
                else :
                    steer_interpolate = np.linspace(self.inter_steer, steer, 1)


                
                for s in steer_interpolate:
                    temp = Actuator()
                    temp.accel = accel /1.3
                    temp.steer = s
                    temp.brake = 0
                    temp.is_waypoint = 1
                    self.actuator_pub.publish(temp)
                    rate.sleep()
                
                
                
                self.inter_steer = steer    
                light = Float32()
                light.data = 0
                self.light_pub.publish(light) 

                rospy.loginfo(f"target steering: {target_steering}")
                
                self.global_waypoints_x = None
                self.global_waypoints_y = None
                

            

            else:
                # rospy.logwarn("No waypoints or current_point available. Skipping loop.")
                rate.sleep()
                continue

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('control_node')
    start = Start()

    rospy.sleep(0.1)  # 퍼블리셔 초기화 대기용
    start.lc_status_pub.publish(String(data="No lanechange command"))

    try:
        start.run_control_loop()
    except rospy.ROSInterruptException:
        pass