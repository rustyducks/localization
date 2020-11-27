import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np
from filterpy.kalman import JulierSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints

import scipy

import signal
import sys
import time
import redis
import threading
from numpy import linalg as la

import arucolvert

import threading


from math import tan, sin, cos, sqrt, atan2


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def move(x, dt):
    return x

def normalize_angle(x):
    while x >= np.pi:
        x -= 2*np.pi
    while x < -np.pi:
        x += 2*np.pi
    return x

def residual_h(a, b):
    y = a - b
    if len(y) != 3:
        # Beacon measurement
        # data in format [dist_1, bearing_1, dist_2, bearing_2,...]
        for i in range(0, len(y), 2):
            y[i + 1] = normalize_angle(y[i + 1])
        #print("residual", a, b, y)
    else:
        y[2] = normalize_angle(y[2])
    return y

def residual_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    #print("y", y)
    return y



def Hx(x, measure_type, landmarks=None):
    """ takes a state variable and returns the measurement
    that would correspond to that state. """
    if measure_type == 0:
        hx = []
        for lmark in landmarks:
            px, py = lmark
            dist = sqrt((px - x[0])**2 + (py - x[1])**2)
            angle = atan2(py - x[1], px - x[0])
            hx.extend([dist, normalize_angle(angle - x[2])])
        #print("hx", hx)
        return np.array(hx)
    elif measure_type == 1:
        return np.array(x)



def state_mean(sigmas, Wm):
    x = np.zeros(3)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = normalize_angle(atan2(sum_sin, sum_cos))
    return x

def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)
    if z_count != 3:
        # Beacon measurement
        for z in range(0, z_count, 2):
            sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
            sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))

            x[z] = np.sum(np.dot(sigmas[:,z], Wm))
            x[z+1] = normalize_angle(atan2(sum_sin, sum_cos))
    else:
        # Direct pose measurement
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        x[2] = normalize_angle(atan2(sum_sin, sum_cos))
    return x



def sqrt_func(x):
    result = None
    try:
        result = scipy.linalg.cholesky(x)
    except np.linalg.LinAlgError:
        x = nearestPD(x)
        result = scipy.linalg.cholesky(x)
    #print("result", result)


    return result



class PubMeasurements(threading.Thread):
    def __init__(self, measurements):
        threading.Thread.__init__(self)
        self.r = redis.Redis(host='localhost', port=6379, db=0)
        self.measurements = measurements

    def run(self):
        self.pub_measurements()

    def pub_measurements(self):
        for m in self.measurements:
            # values = []
            # for c in m:
            #     if c is None:
            #         values += ["None"]
            #     else:
            #         values += [",".join(map(str, c))]
            value = ";".join([",".join(map(str, c)) if c is not None else "None" for c in m])
            self.r.publish("beacons/measurements", value)
            print("publish")
            time.sleep(0.21)

stop = False


def on_beacons_measurements(message):
    data = message['data']
    beacons = []
    measurements = []
    for i, m in enumerate(data.split(b";")):
        if m == b"None":
            continue
        else:
            beacons.append(landmarks[i])
            measurements.extend(list(map(float, m.split(b","))))
    #print("b", beacons)
    ukf.update(measurements, R=np.diag([5 ** 2,
                     0.05 ** 2] * len(beacons)), measure_type=0, landmarks=beacons)

def arucol_reader_thread(ukf):
    arucol = arucolvert.ArucolVert("/dev/ttyUSB3", 115200)
    while not stop:
        x, y, theta = arucol.next_pose()
        print("Updating ukf with", x, y, theta)
        ukf.update(np.array([x, y, theta]), R=np.diag([5 ** 2, 5**2, 0.05**2]), measure_type=1)




dt = 1.0
wheelbase = 0.5

from filterpy.stats import plot_covariance_ellipse

def stop_ukf(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, stop_ukf)

epoch = 0


def create_ukf(
        cmds, landmarks, sigma_vel, sigma_steer, sigma_range,
        sigma_bearing, ellipse_step=1, step=10):

    points = MerweScaledSigmaPoints(n=3, alpha=0.03, beta=2., kappa=0,
                                    subtract=residual_x, sqrt_method=sqrt_func)
    ukf = UKF(dim_x=3, dim_z=2 * len(landmarks), fx=move, hx=Hx,
              dt=dt, points=points, x_mean_fn=state_mean,
              z_mean_fn=z_mean, residual_x=residual_x,
              residual_z=residual_h)

    ukf.x = np.array([203.0, 1549.2, 1.34])
    ukf.P = np.diag([100., 100., .5])
    ukf.R = np.diag([sigma_range ** 2,
                     sigma_bearing ** 2] * len(landmarks))
    ukf.Q = np.diag([10.**2, 10.**2, 0.3**2])

    return ukf

def redis_send_pose(pose):
    pipe = pose_redis.pipeline()
    pipe.set("robot_pose/x", pose[0])
    pipe.set("robot_pose/y", pose[1])
    pipe.set("robot_pose/theta", pose[2])
    pipe.execute()

def run_localisation(ukf, landmarks):
    global epoch
    plt.figure()

    sim_pos = ukf.x.copy()

    # plot landmarks
    if len(landmarks) > 0:
        plt.scatter(landmarks[:, 0], landmarks[:, 1],
                    marker='s', s=60)

    track = []
    while not stop:
        print("epoch:", epoch)
        #print("cov", ukf.P)
        print("pos:", ukf.x)
        epoch += 1
        ukf.predict()
        track.append((time.time(), ukf.x))
        redis_send_pose(ukf.x)
        if epoch % 5 == 0:
            plot_covariance_ellipse(
                (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                facecolor='g', alpha=0.8)
        time.sleep(0.5)
    track = np.array(track)
    #plt.plot(track[:, 0], track[:, 1], color='k', lw=2)
    #plt.axis('equal')
    #plt.title("UKF Robot localization")
    #plt.show()
    with open("track.txt", "w") as f:
        f.write("timestamp, x, y, theta\n")
        for t, p in track:
            f.write("{}, {}, {}, {}\n".format(t, *p))
    print(track)
    return ukf


landmarks = np.array([[0., 0.], [0., 950.], [1360., 450.]])
dt = 0.1
wheelbase = 0.5
sigma_range = 0.3
sigma_bearing = 0.1

measurements = [[[475., -1.57], [475., 1.57], None]] * 15
measurements += [[[747., -2.30], None, [741.8, 0.30]]] * 15
measurements += [[None, [1149., 0.0], [300., -np.pi]]] * 20
print(measurements)


def turn(v, t0, t1, steps):
    return [[v, a] for a in np.linspace(
        np.radians(t0), np.radians(t1), steps)]


# accelerate from a stop
cmds = [[v, .0] for v in np.linspace(0.1, 100., 100)]
cmds.extend([cmds[-1]] * 50)

#turn left
v = cmds[-1][0]
cmds.extend(turn(v, 0, 0.2, 4))
cmds.extend([cmds[-1]] * 75)

# # turn right
# cmds.extend(turn(v, 2, -2, 15))
# cmds.extend([cmds[-1]] * 600)
#
# cmds.extend(turn(v, -2, 0, 15))
# cmds.extend([cmds[-1]] * 150)
#
# cmds.extend(turn(v, 0, 1, 25))
# cmds.extend([cmds[-1]] * 100)

print(cmds)

ukf = create_ukf(cmds, landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
        sigma_range=10, sigma_bearing=0.1, step=1,
        ellipse_step=20)



# ukf = run_localization(
#     cmds, landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
#     sigma_range=3, sigma_bearing=1, step=1,
#     ellipse_step=20)
# print('final covariance', ukf.P.diagonal())
beacon_redis = redis.Redis(host='localhost', port=6379, db=0)
beacon_redis_sub = beacon_redis.pubsub()
beacon_redis_sub.subscribe(**{"beacons/measurements": on_beacons_measurements})
beacon_redis_th = beacon_redis_sub.run_in_thread(sleep_time=0.01)

p = threading.Thread(target=arucol_reader_thread, args=(ukf,))
p.start()

#pub_th = PubMeasurements(measurements)
#pub_th.start()
pose_redis = redis.Redis(host='localhost', port=6379, db=0)


ukf = run_localisation(ukf, landmarks)
beacon_redis_th.stop()
p.join()
print("final postition", ukf.x)






