      ParallelBody: A Multi-sensory Interactive Perception Dataset for Intelligent Vehicles
This dataset is in kitti format and our main baseline is built in SFD(https://github.com/LittlePey/SFD), please refer to the SFD configuration environment if needed.

  The ParallelBody dataset is collected by a multimodal data acquisition platform constructed in the laboratory, which is equipped with sensors other than cameras, LiDAR, millimeter wave radar, IMU, and light, vibration, and sound sensors to obtain information about the scene elements, and the dataset contains challenging scenarios, including a variety of road conditions, lighting conditions, and other challenging scenarios. The sensor setup is as follows:

  
<img src="https://github.com/BUCT-IUSRC/Dataset__ParallelBody/blob/main/readme_image/1.png">


  One of the camera and LiDAR configurations is as follows:
<img src="https://github.com/BUCT-IUSRC/Dataset__ParallelBody/blob/main/readme_image/2.png">


  We divided the dataset scenarios into two main categories: campus scenarios (simple scenarios) and downtown scenarios (complex scenarios), and within each scenario we collected data on a variety of road and lighting conditions, and the data allocation for different scenarios is as follows:
<img src="https://github.com/BUCT-IUSRC/Dataset__ParallelBody/blob/main/readme_image/3.png">


  The total duration is about 2h, and the preliminary estimation is that 150 to 200 20s sequences can be divided, with 68 frames per sequence, totaling over 10,000 frames.

  
  The collected data is divided into two parts: on campus and in the city, where honking is prohibited, there are more speed bumps, and the vehicle speed is around 20km/h. In the city, the speed is around 30-50km/h, with fewer speed bumps and more intersections such as traffic lights.

  
  Data labeling:Example of vibration sensor data

  
<img src="https://github.com/BUCT-IUSRC/Dataset__ParallelBody/blob/main/readme_image/4.png">


  The e8:cb:ed:5a:54:12 device is located at the right front of the vehicle, the e8:cb:ed:5a:53:cc device is located at the right rear of the vehicle, the 38:1e:c7:e3:a3:85 device is located at the left front of the vehicle, and the e8:cb:ed:5a:57:4d device is located at the left rear of the vehicle.
  Example of Light Sensor Data:


<img src="https://github.com/BUCT-IUSRC/Dataset__ParallelBody/blob/main/readme_image/5.png">


  Light sensor: light sensor using Ji Ou speed light illuminance sensor, the relevant information download site is as follows: www.hbousu.com.
  
  
  As the highest value of midday light in about 70,000 Lux, night light is basically below 150, so the plan is in accordance with the 0-200 for the first level, 200-30,000 for the second level, more than 30,000 labeled as the third level of intensity, at the same time, through the recording to assist in the labeling of the passing under the bridge, passing under the building and so on.

  
  Vibration sensor: vibration sensor model WTVB01-BT50, equipment-related information download website: https://wit-motion.yuque.com/wumwnr/docs/ufvs0fdz7d51ow8x.

  
  Since the z-axis vibration displacement after passing a speed bump is around 200μm, and the norm is basically below 20, it is planned to follow 0-25 as the first level, 25-100 as the second level, and more than 100 as the third level of intensity, and at the same time, it is labeled by the audio recording to assist in the cases of passing a speed bump, passing a pothole with the left front wheel, etc. The sound sensor and imu are not labeled.


Sound sensors and imu are not labeled, direct disclosure of raw data.

  Camera laser and millimeter wave radar in accordance with the laser radar labeling data shall prevail, from the laser radar projection to the camera and millimeter wave radar data.
Download:

  The dataset is available from here: http://iusrc.com/column/znsjj
  
  If you have any problem about this work,please feel free to reach us out at zhangtz@mail.buct.edu.cn
