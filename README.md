![ANAT Logo](assets/ANAT_logo.png)

# **Speed Detection and ANPR Integration Project**

## **Overview**
This project integrates **vehicle speed detection** using CCTV footage with an **Automatic Number Plate Recognition (ANPR)** system. It aims to detect speeding vehicles and capture their license plates using advanced computer vision techniques. By leveraging 3D camera calibration, vehicle tracking, and real-time processing, the system provides accurate vehicle speed estimation and automatic license plate detection.

The project is designed for use in traffic management, law enforcement, and other smart city applications where it is crucial to detect speed violations in real-time and maintain records for future reference.

## Why ANAT?

The system is named **ANAT** (أَناة), a word derived from Arabic meaning **patience** and **deliberation**. The choice of this name reflects the broader goal of the project, which is to encourage drivers to slow down, be more patient, and ultimately drive more safely.

Just as **أَناة** signifies taking actions thoughtfully and calmly, this system aims to instill these values in drivers by enforcing speed regulations. The hope is that through the presence of this technology, drivers will adopt safer habits, reducing the risks of speeding and improving road safety overall. In Arabic, **أَناة** carries multiple connotations:

- **أَنْجَزَ عَمَلَهُ في أَناةٍ**: completing work with patience and care.
- **طَويلُ الأَناةِ**: being long-suffering and patient.
- **إِنَّهُ لَذُو أَناةٍ وَرِفْقٍ**: possessing calmness and gentleness.

This name encapsulates the essence of what the system stands for—a technology that doesn't just punish violators but encourages a more composed, deliberate approach to driving.


## **Key Features**
- **Real-time vehicle detection and speed estimation** using computer vision techniques.
- **Automatic Number Plate Recognition (ANPR)** for vehicles exceeding speed limits.
- **Google Cloud Storage (GCS) Integration** for storing images and results securely.
- **3D Bounding Box Construction** for accurate speed measurement using monocular cameras.
- **Fully containerized with Docker** for easy deployment across different environments.

---

## **Technologies and Tools**

- **OpenCV**: Used for video processing, image transformations, and camera calibration.
- **YOLO**: Pre-trained model for real-time object detection, specifically for detecting vehicle license plates.
- **Google Cloud Storage (GCS)**: For storing the results and images securely in the cloud.
- **NumPy**: Handles numerical computations, especially for 3D bounding box calculations and vehicle tracking.
- **Docker**: Used to containerize the application for consistent deployment across different systems.
- **Python**: The main language for development, using its ecosystem for image processing, machine learning, and cloud integration.

---

## **System Components**

### 1. **Vehicle Speed Detection**
- **Camera Calibration**: Ensures that the video stream from CCTV cameras is correctly mapped for distance and speed calculation.
- **Vehicle Detection**: Detects vehicles in each frame using a pre-trained model.
- **3D Bounding Box Estimation**: 3D bounding boxes are constructed for detected vehicles to provide more accurate speed estimation.
- **Speed Calculation**: Tracks vehicles across frames and calculates their speeds based on changes in their positions over time.
  
### 2. **ANPR (Automatic Number Plate Recognition)**
- **License Plate Detection**: The system uses YOLO to detect license plates from the video frames.
- **License Plate Recognition**: Once detected, the license plate numbers are extracted using image processing and OCR techniques.

### 3. **Google Cloud Storage Integration**
- **Secure Data Storage**: The detected vehicle images, license plate details, and speed records are stored in Google Cloud Storage.
- **Organized Data Storage**: Results are stored in organized directories for easy access and analysis.

---

## **Project Structure**

/ (root directory) │ ├── src/ # Source code for the project │ ├── anpr_module/ # ANPR specific module files │ ├── speed_detection/ # Speed detection specific module files │ └── main.py # Main file to run the pipeline │ ├── config/ # Configuration files │ └── config.yaml # YAML config file for the pipeline │ ├── docker/ # Docker-related files │ └── Dockerfile # Docker configuration for the project │ ├── data/ # Datasets and dataset links │ ├── results/ # Stores results or references to GCS results │ ├── tests/ # Unit and integration tests for the project │ ├── LICENSE # License for the repository │ ├── README.md # Project documentation (this file) │ └── requirements.txt # List of required dependencies


---

## **Datasets Used and Licenses**

### 1. **BrnoCompSpeed Dataset**
- **License**: BrnoCompSpeed is a publicly available dataset, but its licensing information should be consulted on its official website.
- **Citation**:
  - Title: *BrnoCompSpeed: Review of Traffic Camera Calibration and Comprehensive Dataset for Monocular Speed Measurement*.
  - Authors: M. F. M. Camargo, V. Vacek, J. Sochor, and A. Herout.
  - Link: [BrnoCompSpeed](https://www.researchgate.net/publication/313879450_BrnoCompSpeed_Review_of_Traffic_Camera_Calibration_and_Comprehensive_Dataset_for_Monocular_Speed_Measurement)

### 2. **Rodo-sol-ANPR Dataset**
- **License**: Refer to the Rodo-sol-ANPR dataset provider for specific licensing terms.
- **Citation**:
  - R. Laroca, E. V. Cardoso, D. R. Lucio, V. Estevam, and D. Menotti, ''On the Cross-dataset Generalization in License Plate Recognition'' in International Conference on Computer Vision Theory and Applications (VISAPP), pp. 166-178, Feb 2022.
  - [SciTePress] [arXiv] [PDF] [BibTeX]

### 3. **KITTI Vision Benchmark Suite**
- **License**: The KITTI dataset is available under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](http://www.cvlibs.net/datasets/kitti/index.php).
- **Citation**:
  - Authors: A. Geiger, P. Lenz, C. Stiller, and R. Urtasun.
  - Description: This dataset includes stereo camera images, LiDAR point clouds, 3D bounding boxes, and calibrated camera data, which are useful for object detection, tracking, and speed estimation.
  - Link: [KITTI Dataset](https://www.tensorflow.org/datasets/catalog/kitti)

---

## **Research Papers Used**

1. **BrnoCompSpeed Review**:
   - *BrnoCompSpeed: Review of Traffic Camera Calibration and Comprehensive Dataset for Monocular Speed Measurement*.
   - [Link](https://www.researchgate.net/publication/313879450_BrnoCompSpeed_Review_of_Traffic_Camera_Calibration_and_Comprehensive_Dataset_for_Monocular_Speed_Measurement)

2. **A Unified Approach to Multi-Camera Object Detection**:
   - [Link](https://arxiv.org/pdf/2003.13137)

3. **Multi-Target Tracking and Detection using Vision and LiDAR**:
   - [Link](https://link.springer.com/epdf/10.1007/s00138-020-01117-x)

4. **Traffic Camera Calibration via Vehicle Vanishing Point Detection**:
   - [Link](https://www.researchgate.net/publication/354499458_Traffic_Camera_Calibration_via_Vehicle_Vanishing_Point_Detection)

5. **Single-Camera Vehicle Speed Measurement**:
   - Authors: V. Gnanasekaran, A. Loganathan.
   - [Link](https://www.researchgate.net/publication/271546929_Single_camera_vehicles_speed_measurement)

6. **Speed Detection Using Image Processing**:
   - [Link](https://ieeexplore.ieee.org/abstract/document/7863015/metrics#metrics)

7. **Speed Violations Tracking via CCTV**:
   - [Link](https://ieeexplore.ieee.org/abstract/document/8710854)

8. **Detection of 3D Bounding Boxes of Vehicles Using Perspective Transformation for Accurate Speed Measurement**:
   - [Link](https://paperswithcode.com/paper/detection-of-3d-bounding-boxes-of-vehicles)

9. **Optical Speed Measurement via Inverse Perspective Mapping**:
   - *Optical Speed Measurement via Inverse Perspective Mapping*.
   - [Link](https://www.researchgate.net/publication/356251967_Optical_Speed_Measurement_via_Inverse_Perspective_Mapping)

---

## **Installation and Setup**

### **Prerequisites**
- **Python 3.8+**
- **Google Cloud Account** for GCS integration
- **Docker** for containerized deployment

### **Installation Steps**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/speed-anpr.git
   cd speed-anpr
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Set up your Google Cloud Storage (GCS) bucket**:
Create a GCS bucket and obtain the service account credentials as described in the Google Cloud Storage Documentation.
Download the service account key file and update your environment variables accordingly.

4.**Update the config.yaml file**:
  ```bash
  video_path: 'path/to/your/video.mp4'
  LP_model: 'path/to/your/yolo_model.pt'
  calibration_file: 'path/to/camera_calibration.json'
  road_mask_file: 'path/to/road_mask.npy'
  gcs_bucket_name: 'your-gcs-bucket'
  detection_confidence: 0.4
  speed_threshold: 80
```
5.**Run the project**:
  ```bash
  python src/main.py --config config/config.yaml
  ```
--

## **Running the Project with Docker**

1.**Build the Docker image**:
 ```bash
  docker build -t speed-anpr-app .
```
2.**Run the Docker container:**:
 ```bash
  docker run -e GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json" speed-anpr-app 
```
## **License**
This project is licensed under the MIT License - see the LICENSE file for details.

