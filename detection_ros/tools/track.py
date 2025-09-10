#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray
import math

from detection_ros.msg import ObstacleDetection

import time
import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment

import glob
from pathlib import Path

# import mayavi.mlab as mlab
import numpy as np
import scipy.linalg as linalg

import sys
sys.path.append("/home/wyt/code/test_git/src/detection_ros") #修改为自己的路径

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# 卡尔曼滤波器类（仅平滑位置）
class KalmanFilter:
    def __init__(self, dt=0.1):
        self.state_dim = 3
        self.measure_dim = 3
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1.0
        self.F = np.eye(self.state_dim)
        self.H = np.eye(self.measure_dim, self.state_dim)
        self.Q = np.eye(self.state_dim) * 0.01
        self.R = np.eye(self.measure_dim) * 0.1

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def correct(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

# 跟踪对象类
class Tracker:
    def __init__(self, detection, tracked_id, dimensions, yaw, label, dt=0.1):
        self.kf = KalmanFilter(dt=dt)
        self.kf.x = detection  # 初始位置 [x, y, z]
        self.tracked_id = tracked_id
        self.missing_match = 0
        self.dimensions = dimensions  # 目标尺寸 [w, l, h]
        self.yaw = yaw  # 目标朝向
        self.label = label  # 目标类别
        self.history_positions = [detection]  # 存储历史位置
        self.history_timestamps = [rospy.Time.now()]  # 存储历史时间戳
        self.dt = dt

    def predict(self):
        self.kf.predict()

    def update(self, detection, dimensions, yaw, label, timestamp):
        self.kf.correct(detection)
        self.missing_match = 0
        self.dimensions = dimensions
        self.yaw = yaw
        self.label = label
        self.history_positions.append(self.kf.x.copy())
        self.history_timestamps.append(timestamp)  # 记录实际时间戳
        if len(self.history_positions) > 2:
            self.history_positions.pop(0)
            self.history_timestamps.pop(0)

    def get_speed(self):
        if len(self.history_positions) < 2:
            return 0.0
        pos_t = self.history_positions[-1][:2]  # 当前帧 [x, y]
        pos_t_minus_1 = self.history_positions[-2][:2]  # 前一帧 [x, y]
        delta = pos_t - pos_t_minus_1
        time_diff = (self.history_timestamps[-1] - self.history_timestamps[-2]).to_sec()  # 实际时间差
        if time_diff <= 0:  # 防止除以零
            return 0.0
        speed = np.linalg.norm(delta) / time_diff
        # speed = np.linalg.norm(delta) / self.dt
        return speed

# 数据关联函数
def associate_detections_to_trackers(detections, trackers):
    if not trackers:
        return [], list(range(len(detections))), []
    
    # 如果 detections 是 NumPy 数组，检查其形状
    if isinstance(detections, np.ndarray):
        if detections.shape[0] == 0:  # 检查数组是否为空
            return [], [], list(range(len(trackers)))
    # 对于非张量类型（如列表），保持原有逻辑
    else:
        return [], [], list(range(len(trackers)))

    cost_matrix = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            cost_matrix[d, t] = np.linalg.norm(det - trk.kf.x)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = list(zip(row_ind, col_ind))
    unmatched_detections = [d for d in range(len(detections)) if d not in row_ind]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in col_ind]
    return matches, unmatched_detections, unmatched_trackers

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path: 根目录
            dataset_cfg: 数据集配置
            class_names: 类别名称
            training: 训练模式
            logger: 日志
            ext: 扩展名
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        

class Detection_ROS:
    def __init__(self):
        # 标签映射字典
        self.label_to_category = {
            1: 'big_boat',
            2: 'medium_boat',
            3: 'small_boat',
            4: 'sign',
            5: 'piers',
            6: 'lighthouse',
            7: 'unknown'
        }
        config_path, ckpt_path = self.init_ros()
        self.init_detection(config_path, ckpt_path)
        self.trackers = []  # 跟踪器列表
        self.tracked_id = 0  # 跟踪 ID 计数器
        self.pub_markers = rospy.Publisher('/tracked_markers', MarkerArray, queue_size=10)  # 发布 MarkerArray


    def init_ros(self):
        """ Initialize ros parameters """

        config_path = rospy.get_param("/config_path", "/home/wyt/code/test_git/src/detection_ros/tools/cfgs/custom_models/dsvt_maritime.yaml") #模型配置文件路径
        ckpt_path = rospy.get_param("/ckpt_path", "/home/wyt/code/checkpoint_epoch_80.pth") #模型权重文件路径

        return config_path, ckpt_path


    def init_detection(self, config_path, ckpt_path):
        """ Initialize second model """
        logger = common_utils.create_logger() # 创建日志
        logger.info('-----------------Quick Demo of Detection-------------------------')
        cfg_from_yaml_file(config_path, cfg)  # 加载配置文件
        
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            ext='.bin', logger=logger
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        # 加载权重文件
        self.model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
        self.model.cuda() # 将网络放到GPU上
        self.model.eval() # 开启评估模式


    def rotate_mat(self, axis, radian):
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        return rot_matrix
    
    def publish_markers(self, trackers, detections, dimensions, labels, frame_id, matches):
        #可视化追踪文本
        marker_array = MarkerArray()

        # 删除所有旧的 Marker
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = rospy.Time.now()
        marker_array.markers.append(delete_marker)

        # print(f"Publishing markers for {len(detections)} detections with {len(trackers)} trackers")
        detection_to_tracker = {d_idx: t_idx for d_idx, t_idx in matches}
        for d_idx in range(len(detections)):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "tracked_objects"
            marker.id = d_idx
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            # 使用检测结果的位置，而不是跟踪器的位置
            marker.pose.position.x = detections[d_idx][0]
            marker.pose.position.y = detections[d_idx][1]
            marker.pose.position.z = detections[d_idx][2] + dimensions[d_idx][2] / 2  # 放置在目标上方
            marker.pose.orientation.w = 1.0
            marker.scale.z = 5.0  # 增大文本高度
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            # 如果检测框匹配到跟踪器，使用跟踪器的速度和 ID
            if d_idx in detection_to_tracker:
                t_idx = detection_to_tracker[d_idx]
                trk = trackers[t_idx]
                speed = trk.get_speed()
                tracked_id = trk.tracked_id
                category = self.label_to_category.get(labels[d_idx], 'unknown')
                marker.text = f"{category}:{speed:.2f} m/s"
                # marker.text = f"Tracker {tracked_id} ({category}): Speed {speed:.2f} m/s"
            else:
                # 未匹配的检测框，显示默认文本
                category = self.label_to_category.get(labels[d_idx], 'unknown')
                marker.text = f"Untracked {category}"
                # marker.text = f"Untracked ({category}): Speed 0.00 m/s"
            # print(f"Marker {d_idx}: Position = [{marker.pose.position.x}, {marker.pose.position.y}, {marker.pose.position.z}], Text = {marker.text}")
            marker_array.markers.append(marker)

        self.pub_markers.publish(marker_array)

    def publish_det(self, trackers, detections, dimensions, yaws, labels, frame_id, matches):

        obj = ObstacleDetection()
        obj.header.frame_id = frame_id
        obj.header.stamp = rospy.Time.now()

        obstacle_type = []
        obstacle_id = []
        obstacle_l = []
        obstacle_w = []
        obstacle_x = []
        obstacle_y = []
        obstacle_v = []
        obstacle_q = []

        detection_to_tracker = {d_idx: t_idx for d_idx, t_idx in matches}
        
        for d_idx in range(len(detections)):
            # 使用检测结果的位置，而不是跟踪器的位置
            obstacle_l.append(dimensions[d_idx][0])
            obstacle_w.append(dimensions[d_idx][1])
            obstacle_x.append(detections[d_idx][0])
            obstacle_y.append(detections[d_idx][1])
            obstacle_q.append(math.degrees(yaws[d_idx]))

            # 如果检测框匹配到跟踪器，使用跟踪器的速度和 ID
            if d_idx in detection_to_tracker:
                t_idx = detection_to_tracker[d_idx]
                trk = trackers[t_idx]
                speed = trk.get_speed()
                obstacle_v.append(speed)
                tracked_id = trk.tracked_id
                obstacle_id.append(tracked_id)
                obstacle_type.append(labels[d_idx])
            else:
                obstacle_v.append(-1)
                obstacle_id.append(-1)
                obstacle_type.append(-1)

        obj.obstacle_type = obstacle_type
        obj.obstacle_id = obstacle_id
        obj.obstacle_l = obstacle_l
        obj.obstacle_w = obstacle_w
        obj.obstacle_x = obstacle_x
        obj.obstacle_y = obstacle_y
        obj.obstacle_v = obstacle_v
        obj.obstacle_q = obstacle_q

        pub_det.publish(obj)
        rospy.loginfo(obj)


    def lidar_callback(self, msg):
        """ Captures pointcloud data and feed into second model for inference """
        pcl_msg = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z","intensity","ring"))
        np_p = np.array(list(pcl_msg), dtype=np.float32)
        # 旋转轴
        #rand_axis = [0,1,0]
        #旋转角度
        #yaw = 0.1047
        #yaw = 0.0
        #返回旋转矩阵
        #rot_matrix = self.rotate_mat(rand_axis, yaw)
        #np_p_rot = np.dot(rot_matrix, np_p[:,:3].T).T

        # convert to xyzi point cloud
        x = np_p[:, 0].reshape(-1)
        y = np_p[:, 1].reshape(-1)
        z = np_p[:, 2].reshape(-1)
        if np_p.shape[1] == 4: # if intensity field exists
            i = np_p[:, 3].reshape(-1)
        else:
            i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        points = np.stack((x, y, z, i)).T
        #print(points.shape)
        # 组装数组字典
        input_dict = {
            'frame_id': msg.header.frame_id,
            'points': points
        }
        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict) # 数据预处理
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict) # 将数据放到GPU上
        pred_dicts, _ = self.model.forward(data_dict) # 模型前向传播
        scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
        mask = scores > 0.5
        scores = scores[mask]
        boxes_lidar = pred_dicts[0]['pred_boxes'][mask].detach().cpu().numpy()
        label = pred_dicts[0]['pred_labels'][mask].detach().cpu().numpy()
        num_detections = boxes_lidar.shape[0]
        #rospy.loginfo("The num is: %d ", num_detections)

        # 提取检测结果
        detections = boxes_lidar[:, :3]  # [x, y, z]
        dimensions = boxes_lidar[:, 3:6]  # [w, l, h]
        yaws = boxes_lidar[:, 6]  # [yaw]

        # 更新跟踪器
        matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, self.trackers)

        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(detections[d_idx], dimensions[d_idx], yaws[d_idx], label[d_idx], msg.header.stamp)

        for d_idx in unmatched_detections:
            self.trackers.append(Tracker(detections[d_idx], self.tracked_id, dimensions[d_idx], yaws[d_idx], label[d_idx]))
            self.tracked_id += 1

        for t_idx in unmatched_trackers:
            self.trackers[t_idx].predict()
            self.trackers[t_idx].missing_match += 1

        self.trackers = [trk for trk in self.trackers if trk.missing_match < 3]

        arr_bbox = BoundingBoxArray()
        for i in range(num_detections):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2])  # + float(boxes_lidar[i][5]) / 2  # 注释掉高度调整
            bbox.dimensions.x = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z = float(boxes_lidar[i][5])  # height
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = label[i]
            arr_bbox.boxes.append(bbox)

        arr_bbox.header.frame_id = msg.header.frame_id

        if len(arr_bbox.boxes) != 0:
            # pub_bbox.publish(arr_bbox) #可视化3D框
            # self.publish_pcd(points, msg.header.frame_id) #可视化对应点云
            # self.publish_markers(self.trackers, detections, dimensions, label, msg.header.frame_id, matches) #可视化追踪文本

            self.publish_det(self.trackers, detections, dimensions, yaws, label, msg.header.frame_id, matches) #发布追踪结果
            

    def publish_pcd(self, cloud, frame_id):
        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = frame_id

        msg_segment = pc2.create_cloud(header = header, fields = _make_point_field(4), points = cloud)
        #msg_segment = pc2.create_cloud(header=header, points=cloud)

        pub_velo.publish(msg_segment)

def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str_('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str_('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str_('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str_('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str_('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]
if __name__ == '__main__':
    global sec
    sec = Detection_ROS()

    rospy.init_node('detection_ros_node', anonymous=True)

    # subscriber
    sub_velo = rospy.Subscriber("/merged_point_cloud", PointCloud2, sec.lidar_callback, queue_size=1,
                                     buff_size=2 ** 12)

    # publisher
    pub_velo = rospy.Publisher("/modified", PointCloud2, queue_size=1)
    pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=10)
    pub_det = rospy.Publisher("/obstacle_detection", ObstacleDetection, queue_size=10)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
