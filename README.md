# maritime_detection
Based on the ros1 system, obtain point cloud data and use the trained target detection model to detect obstacle information (obstacle_type（障碍物类型）、obstacle_id（障碍物ID）、obstacle_l（障碍物长）、obstacle_w（障碍物宽）、obstacle_x（障碍物位置x）、obstacle_y（障碍物位置y）、obstacle_v（障碍物相对速度）、obstacle_q（障碍物相对方向） )<br>

## 运行环境
* Ubuntu20.04<br>
* Nvidia 4080ti 显卡
* cuda12.4<br>
* ros-noetic<br>
* python环境使用conda参考[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)创建并激活<br>

## 运行步骤
### 将代码git clone到本地
```Bash
git clone https://github.com/wyt1004/maritime_detection.git
```
### 编译
```Bash
# 进入搭建好的openpcdet环境
conda activate openpcdet
pip install --user rospkg catkin_pkg
pip install pyquaternion

#安装对应ros版本的包
sudo apt-get install ros-noetic-pcl-ros
#rviz可视化需要的包
sudo apt-get install ros-noetic-jsk-recognition-msg
sudo apt-get install ros-noetic-jsk-rviz-plugins

#编译ros代码，在最外层文件夹（maritime_detection）下执行
catkin_make
```
### 运行
#### 下载目标检测模型文件
[下载链接](https://drive.google.com/file/d/1AGP6TP3bOziVDfPfhNwEhy60hAsb9fEo/view?usp=drive_link)<br>

修改track.py代码中配置文件和模型文件的路径，改为当前系统下的路径。<br>

### 启动点云拼接程序
下载[publish_combined_lidar](https://github.com/wyt1004/publish_combined_lidar),编译成功后运行，成功发布`/merged_point_cloud`后为运行成功。

#### 运行程序
执行以下命令运行：
```Bash
roslaunch detection_ros pointpillars.launch
```
### 输出
当终端中有如下内容输出，则表示运行成功。订阅话题 `/obstacle_detection` 则可以接收检测信息。
```Bash
[INFO] [1745371901.085010]: header: 
  seq: 0
  stamp: 
    secs: 1745371901
    nsecs:  84680795
  frame_id: "laser_link"
obstacle_type: 
  - 2
  - 3
  - 3
  - 3
  - 1
obstacle_id: [0, 2, 3, 1, 8]
obstacle_l: 
  - 36.91796
  - 17.552639
  - 23.67039
  - 17.71622
  - 48.93801
obstacle_w: 
  - 7.513003
  - 3.8569925
  - 5.5574965
  - 5.942371
  - 9.364847
obstacle_x: 
  - 112.384674
  - 106.68552
  - 123.25568
  - 112.30313
  - -101.5931
obstacle_y: 
  - 61.527557
  - 58.552505
  - 87.456955
  - -8.589149
  - 180.41315
obstacle_v: 
  - 6.667125333484149
  - 6.835538087586001
  - 6.949031536973027
  - 6.7081580126351765
  - 0.0
obstacle_q: [80.3747428317453, 79.45003767096311, 92.44488237826258, 88.58348274232006, -109.5875370439707]
```
