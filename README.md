<h1 align="center">HUMAN-EMOTION-RECOGNITION</h1>


# PROBLEM STATEMENT		
The goal of this project is to develop a computer vision system that can accurately recognize and classify human emotions based on facial expressions.			
 <h2>OBJECTIVE</h2>
<pre>•Collect and preprocess a large dataset of facial images with labeled emotions.
•Train and fine-tune deep learning models to accurately recognize and classify emotions.
•Evaluate the performance of the developed system on benchmark datasets and real-world scenarios.
•Optimize the system for real-time emotion recognition applications.
•Explore potential applications of the developed system in various fields, such as mental health, human-computer interaction, and marketing.</pre>

---

# DATASET DETAIL [DRIVE URL]			

<h3>Image Data  - download and upload in a link</h3>
•	Dataset Description: Includes images of human faces labeled with different emotions. Commonly used datasets are AffectNet, RAF-DB, and extended Cohn-Kanade.
•	Link: 
https://amritavishwavidyapeetham-my.sharepoint.com/:f:/g/personal/cb_en_u4cse21155_cb_students_amrita_edu/EuVyoWAk94NLjHYdmK7HBrUBT69syVLVyssW_86AXKgszw?e=7eNo3g

<h3>Video Data - download and upload in a link</h3>
•	Dataset Description: Contains video clips capturing facial expressions in various emotional states. Each video is labeled with the corresponding emotion.
•	Link: 
https://amritavishwavidyapeetham-my.sharepoint.com/:f:/g/personal/cb_en_u4cse21155_cb_students_amrita_edu/Ek_pRd7Egq1GkiscwohKMlgBjmoKbKQuCzvmjA-0sJj-BA?e=IuW9W2

---
# RELATED WORK			 
    
<h3>PAPER –1 :</h3>
Facial emotion recognition using convolutional neural networks (FERC) 

The paper introduces a novel technique called Facial Emotion Recognition using Convolutional Neural Networks (FERC) aimed at detecting five key facial expressions: displeasure/anger, sad/unhappy, smiling/happy, feared, and surprised/astonished. The methodology involves a two-part CNN framework: one part for background removal and the other for extracting facial feature vectors using an expressional vector (EV). The FERC model is tested on diverse datasets, including a 10,000-image database and extended Cohn-Kanade dataset, to ensure comprehensive validation. The paper emphasizes FERC's improved accuracy over single-level CNN approaches and highlights potential applications in predictive learning, lie detection, human-computer interaction, and more. Future research aims to enhance the model's capabilities and explore further applications.

<h3>PAPER – 2 :</h3>
A study on computer vision for facial emotion recognition 

This study investigates the application of computer vision for facial emotion recognition using a deep neural network (DNN) that integrates convolutional neural networks (CNN), squeeze-and-excitation networks, and residual neural networks. By training on the AffectNet and Real-World Affective Faces Database (RAF-DB) datasets, the model aims to improve the accuracy of emotion recognition by identifying critical facial features. The methodology incorporates attention mechanisms and class activation mapping to enhance discriminative feature learning. Compared to existing methods, the proposed approach shows competitive results, especially with the use of transfer learning. Future research directions include improving attention mechanisms, addressing practical deployment challenges, and further exploring facial landmarks' role in emotion recognition.

 <h3>PAPER -3:</h3>
 Facial Expression Recognition Using Computer Vision: A Systematic Review 

This systematic review provides a comprehensive analysis of research in facial expression recognition (FER) from January 2006 to April 2019. It examines 112 papers to identify commonly used methods and algorithms, including face detection, feature extraction techniques like PCA, LBP, and Gabor filters, and classification algorithms such as SVMs and CNNs. The review highlights the challenge of maintaining high accuracy in uncontrolled and pose-variant scenarios compared to controlled environments. It also emphasizes the importance of developing robust multimodal systems to handle real-world complexities. Future research directions include improving FER systems' robustness and generalization, addressing challenges in uncontrolled environments, and exploring new techniques and datasets.

<h3>PAPER – 4: </h3>
Emotion Recognition Using Feature-level Fusion of Facial Expressions and Body Gestures 

This study explores automatic emotion recognition by integrating facial expressions and body gestures, aiming to enhance accuracy in understanding human emotional behavior. It develops the Amrita Emotion Database-2 (AED-2), featuring subjects expressing seven basic emotions through both facial and body gestures. The proposed model uses feature-level fusion to combine data from facial expressions and upper body poses, employing techniques such as skin segmentation and histogram equalization for feature extraction. Experiments demonstrate improved accuracy with fusion using classifiers like BayesNet and SVM. The paper concludes by underscoring the benefits of combining modalities and suggests future research directions to further refine emotion recognition systems.

<h3>PAPER 5: </h3>
A Comparative Study on different approaches of Real Time Human Emotion Recognition based on Facial Expression Detection 
In this paper, a comparative study of several approaches to real-time facial expression recognition for human emotion identification was carried out, which determines the relevance. A review of the best studies of feature extraction and classification is presented, which plays the most significant role in reevaluating system accuracy. Moreover, the problems of various emotion classification models, for example, Ekman’s basic emotions or Russell’s circumplex model and the best related work in the field were discussed. Based on these studies and their results, comparisons of the strengths and weaknesses of different algorithms for real-time emotion recognition on facial expressions were offered for opportunities for system advancements. The conclusion summarizes the importance of choosing the right features and classifiers and the further development of real-time emotion recognition systems.

--- 
# METHODOLOGY

Method Overview: Human Emotion Recognition Using Convolutional Neural Networks (CNN) and Facial Landmark Detection.
	
<h2>Introduction:</h2>
In this project I have tried to implement an emotion recognition algorithm that uses Convolutional Neural Networks (CNN) for facial landmark detection and another CNN layer for classification of human emotions based on image of the face. Vision-based facial landmark detection has been employed for accurate localization of features, which is crucial in detecting emotions and Convolutional Neural Networks (CNNs) have been highly successful deep learning models used in image analysis.

<h3>Image Acquisition:</h3>
High-resolution cameras with optimal frame rates capture clear details of facial expressions.

<h3>Camera Operation:</h3>
Accurate cameras give crystalclear images or for video frames. Resolution & Frame rate for the important parameters of facial expressions are adjusted to sufficiently record clear details.
Datasets: train & validate the model
Grayscale Conversion: Each frame is converted from color (RGB) to grayscale to simplify image data and reduce computational complexity.
Histogram Equalization: Applied to improve the contrast of facial features.
Resizing: Images are resized to a standard dimension (e.g., 48x48 pixels) to ensure uniform input size for the CNN.
<h3>Face Detection:</h3>
Dlib: Histogram of Oriented Gradients (HOG) + Linear SVM or deep learning-based face detection methods for more accuracy.
<h3>Facial Landmark Detection:</h3>
Harris- Corner Detector: Identifies key facial landmarks (e.g., eyes, nose, mouth) to aid in emotion recognition.
Keypoint Detection: Precise localization of landmarks is critical for accurate feature extraction.
<h3>Feature Extraction Using CNN:</h3>
Pre-trained Models: Use of pre-trained CNN models like VGG-Face, ResNet, or MobileNet, fine-tuned for emotion recognition tasks.
<h3>Feature Matching and Analysis:</h3>
Deep Features: Extracted deep features from the CNN represent complex patterns in facial expressions.
Dimensionality Reduction: Techniques like Principal Component Analysis (PCA) or t-SNE are used to reduce feature dimensionality for further analysis.
<h3>Emotion Classification:</h3>
Machine Learning: SVM, Random Forest, or KNN classifiers are trained on extracted features for emotion classification.
Deep Learning: End-to-end training of CNNs with softmax or fully connected layers for final emotion classification.
<h3>Post-Processing:</h3>
Feature Enhancement: Laplacian frequency filtering to enhance edges and fine details of facial features.
Boundary Smoothing: Morphological operations to smooth facial landmark regions, ensuring clear feature representation.
<h3>Emotion Detection and Classification:</h3>
Texture Analysis: Extract texture features such as Haralick features or Local Binary Patterns (LBP) from facial regions to characterize expression properties.
Ensemble Methods: Combining outputs from multiple classifiers or CNN models to improve accuracy and robustness.
<h3>Validation and Evaluation:</h3>
Dataset Division: The dataset is divided into training, validation, and test sets to ensure model generalization.
Performance Metrics: Evaluation using metrics such as accuracy, precision, recall, F1-score, and confusion matrix to assess the model's performance.
<h3>Integration with Real-Time Applications:</h3>
Real-Time Emotion Recognition: Implementing the trained model in real-time systems using webcam inputs for live emotion detection.
Human-Computer Interaction (HCI): Integration with HCI systems for applications like sentiment analysis, adaptive user interfaces, and interactive entertainment.

# CAMERA RELATED PROPERTIES AND ALGORITHMS

1.	Resolution: The number of pixels that make up the image.
Example: A camera with a resolution of 1920x1080 pixels (Full HD) captures more detail than a camera with 640x480 pixels (VGA). Higher resolution is preferred for capturing fine details of facial expressions.
2. Frame Rate: The number of frames captured per second (fps).
Example: A frame rate of 30 fps is typically sufficient for smooth video capture, whereas 60 fps can be used for capturing rapid changes in facial expressions.
3. Shutter Speed: The duration for which the camera's sensor is exposed to light.
Example: A fast shutter speed (e.g., 1/1000 second) can reduce motion blur, which is beneficial when capturing dynamic facial expressions. A slower shutter speed (e.g., 1/30 second) might introduce motion blur but is useful in low-light conditions.
4. Aperture: The size of the opening through which light enters the camera.
Example: A larger aperture (e.g., f/2.0) allows more light to enter, which is useful in low-light conditions but reduces the depth of field. A smaller aperture (e.g., f/8.0) increases the depth of field, keeping more of the image in focus.
5. ISO Sensitivity: The sensitivity of the camera sensor to light.
Example: A low ISO (e.g., ISO 100) is used in bright conditions to reduce noise, while a high ISO (e.g., ISO 3200) is used in low-light conditions but may introduce more noise.
6. White Balance: Adjusts the colors to match the lighting conditions.
Example: Setting the white balance to "Daylight" for outdoor conditions or "Tungsten" for indoor lighting ensures that the colors of the captured images are accurate.
7. Focus: The clarity and sharpness of the captured image.
Example: Auto-focus can be used for dynamic scenes where the subject's distance from the camera changes, while manual focus allows for precise control in static scenes.
8. Field of View (FOV): The extent of the observable area captured by the camera.
Example: A wide-angle lens with a 90-degree FOV can capture more of the scene, which is useful for group emotion recognition, whereas a narrow FOV is better for capturing detailed expressions of a single individual.
9. Dynamic Range: The range of light intensities from the darkest shadows to the brightest highlights that the camera can capture.
Example: A high dynamic range (HDR) camera can capture both bright and dark areas of a scene without losing detail, which is beneficial in varying lighting conditions.
10. Image Stabilization: Reduces blurriness caused by camera motion.
Example: Optical or digital image stabilization can help maintain image clarity when the camera is handheld or subject to minor movements.

# MAGE PROCESSING TECHNIQUES

<h3>->	Gray scale </h3>

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/1a6e8d73-e5ab-4f26-94a8-96b1548388f6)


<h3>->	Morphological Operations– Erosion, Dilation</h3>

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/5ef01472-5c2c-4fc9-a7c8-3721634de6bd)


<h3>->	Histogram equilization</h3>

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/2423ee1a-614e-4af1-9863-72ecb570a356)


<h3>->	Laplacian filter</h3>

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/08ec231a-b102-43c3-81e4-8eccfc8fa3b8)


<h3>->	Harris – Corner Detector</h3>
 
![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/36e36323-6ea9-4559-bd1b-ee545c26747a)

<h3>->	Bi-lateral, Canny Edge Detector</h3>

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/fb15e2a5-6649-4e99-a8fd-80c4313f968c)


# FEATURE DETECTION AND MATCHING
<h3>1. Scale-Invariant Feature Transform (SIFT):</h3> 
SIFT detects and describes local features in images. It is invariant to scale, rotation, and illumination changes, making it suitable for identifying distinct facial features under varying conditions.
Application: Detect key facial landmarks and match these features across different images to ensure consistency in emotion recognition.

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/9611bf79-56db-447a-9134-7d09853133c4)


<h3>2. BRISK (Binary Robust Invariant Scalable Keypoints):</h3>
BRISK is an image feature detection algorithm designed for real-time applications. It is part of the family of binary feature descriptors, known for their computational efficiency and robustness.

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/5e6533a1-d2a2-4858-bbed-ae93ef7f8954)


<h3>3. Oriented FAST and Rotated BRIEF (ORB):</h3>
ORB is an efficient and fast feature detection and description algorithm. It combines the FAST keypoint detector and the BRIEF descriptor, with added orientation robustness.
Application: Apply ORB for efficient feature detection in real-time systems where computational resources are limited.
 
![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/614aaf63-917a-48f0-96ca-166512befde1)

<h3>4. Histogram of Oriented Gradients (HOG):</h3> 
HOG captures edge or gradient structures that are characteristic of local shape. It is often used for object detection, including face detection.
Application: Extract HOG features to represent facial expressions and use these features for emotion classification.

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/2888bb8c-2b3c-4a37-a42c-f6930cbafa5a)


# VIDEO PROCESSING CONCEPTS

<h2>KeyFrame extraction - ></h2>
<h3>OUTPUT:</h3>

<h4>KEYFRAME - 1:</h4>

![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/2ddd1558-dea3-41c3-aa2f-d8b17366d1e7)


<h4>KEYFRAME – 2:</h4>

![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/61191580-1bd7-4001-956c-a6cd19f313f0)

<h4>KEYFRAME – 3:</h4>

![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/ac63c6d3-b4f5-4ec1-b1ad-ff9eba0726cb)

<h4>KEYFRAME – 4:</h4>

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/808fa455-2881-4148-93ef-8d7d808dafd0)

<h4>KEYFRAME – 5:</h4>

![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/544fd2f9-f5da-46dc-95e2-7f26132a6491)

<h2>Optical Flow Algorithm –</h2>

<h3>Sparse Optical Flow</h3>

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/4fc23ffd-7d96-4fae-bf69-036373dadb2c)

<h3>Dense Optical Flow</h3>

 ![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/5dba2182-d0c7-442e-8db3-9db93ac2537f)

<h3>Classifiers for recognition</h3>

![image](https://github.com/Zakeer-official/Human-Emotion-Recognition/assets/102544273/4bedc7a9-6a59-4926-9778-e9ca811a57cf)
 

# CONCLUSION
The project has comprehensively considered computer vision approaches by utilizing convolutional neural networks (CNN) for feature extraction and facial landmark detection, essential for accurate emotion recognition. Key methodologies include preprocessing techniques like grayscale conversion and histogram equalization, face detection using HOG and deep learning methods, and the use of pre-trained CNN models for feature extraction. Additionally, various classifiers such as SVM, Random Forest, and deep learning models have been employed for emotion classification. The integration of these algorithms into real-time applications ensures the system's practical utility in diverse fields such as mental health, human-computer interaction, and marketing. Future work aims to enhance system robustness and explore multimodal approaches combining facial and body gesture data.
