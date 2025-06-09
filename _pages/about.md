---
permalink: /
title: "Yihang Zou"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---


I am an undergraduate student majoring in Artificial Intelligence at South China University of Technology. My primary interests include:

- **Detection and Processing of Human Physiological Signals**: Focusing on the acquisition and analysis of various physiological signals, with a particular emphasis on brain-computer interfaces (BCI) and neurosignal analysis.
- **Classification of Brain Signals**: Utilizing advanced machine learning techniques to classify brain signals from modalities such as EEG (electroencephalogram), fNIRS (functional near-infrared spectroscopy), and fMRI (functional magnetic resonance imaging).
- **Brain Signal Encoding and Decoding**: Exploring the transformation of neural activities into diverse outputs, such as images, videos, text, and speech. This involves developing systems that can interpret and generate meaningful information from brain signals.
- **Temporal Network Models**: Investigating the application of temporal network models to understand the dynamic nature of physiological signals and their underlying patterns.
- **Signal Processing Techniques**: Enhancing the quality and interpretability of neural data through advanced signal processing methods.

---


## Educational Background

**South China University of Technology** (2022 - 2026)  
Major: Artificial Intelligence  
Core Courses: Machine Learning, Deep Learning, Data Structures and Algorithms, Introduction to Artificial Intelligence, Python Programming, Data Analysis, etc.

## Personal Skills

- **Academic Ability**:  
  - GPA: **3.74/4.00**

- **Professional Skills**:  
  - Proficient in programming languages such as Python, C++, C#, R, and Matlab, with a focus on Python.  
  - Experienced in using machine learning and deep learning frameworks, such as PyTorch.  
  - Skilled in data analysis tools, including Pandas, NumPy, and Matplotlib.  
  - Strong capabilities in data analysis, model construction, and algorithm optimization.  
  - In-depth research in brain signal processing, including functional near-infrared spectroscopy (fNIRS), electroencephalogram (EEG), and functional magnetic resonance imaging (fMRI).

- **Language Skills**:  
  - English: 
    - **CET-4 (633)**
    - **CET-6 (581)**
    - **IELTS 7.0**
  - Fluent in speaking, listening, reading, and writing, with the ability to conduct academic communication and read professional literature.

- **Research Abilities**:  
  - Participated in multiple research projects, such as "Affective Computing Based on fNIRS in Virtual Reality Environments" (core member of the National Innovation Project).  
  - Achieved results in fNIRS signal classification, with a related paper submitted to the MICCAI conference.  
  - Co-author of the patent "**A Method and System for Identifying Depression Tendencies and Guiding Mindfulness**" (core member, application under review).

- **Academic Competitions**:  
  - 2023 Baidu Paddle Cup **Second Prize**.  
  - 2024 MCM/ICM (Mathematical Contest in Modeling) **Meritorious Award(top 6%)**.  
  - 2025 Mathorcup **First Prize**(top 10%), proposed to recommend national-level awards.
  - 2024 Greater Bay Area Mathematical Finance Modeling Competition **Second Prize**.  
  - 2024 14th Asia-Pacific Mathematical Contest in Modeling **Third Prize**.  
  - 2024 International Innovation Competition for College Students, Industry Track Final **Bronze Award**.

- **Teamwork and Communication Skills**:  
  - Strong team spirit and communication skills, able to collaborate effectively with members from diverse backgrounds to advance project progress.



---
## Research Project Experience


### Research Project Experience 1: Affective Computing Based on Functional Near-Infrared Spectroscopy in Virtual Reality Environments (National Innovation Project, Hundred-Step Ladder Climbing Plan)

**Project Duration**: June 2023 - June 2024

**Project Description**:  
Participated in a national-level innovation project, focusing on detecting and assisting in the treatment of depression tendencies through virtual reality (VR) technology and functional near-infrared spectroscopy (fNIRS). The project designed and developed four VR scenarios, two for detecting depression tendencies and the other two for mindfulness guidance and auxiliary treatment. By recording the fNIRS signals of the cerebral cortex of subjects in specific VR environments and combining their performance in spatial memory navigation tasks, the project explored the relationship between depression tendencies and cognitive functions.

**Technical Details**:

- Developed VR environments using **Unity**, with **C#** as the programming language.
- Designed and implemented a VR scenario for a city spatial memory and navigation task. Research indicates that the spatial memory navigation ability of the hippocampus is related to the degree of depression, with subjects having depression tendencies performing worse in spatial memory tasks. This scenario indirectly assessed the subjects' depression levels.
- While subjects completed the VR tasks, their cerebral cortex fNIRS signals were recorded to analyze their brain activity patterns and identify neurophysiological indicators related to depression.
- Combined psychological assessment scales to comprehensively evaluate the subjects' emotional states and cognitive functions, verifying the effectiveness of the VR tasks.
- Preprocessed and analyzed the collected fNIRS data to extract brain activity features related to depression, exploring new biomarkers.

**Achievements**:

- The project results have applied for the patent "A Method and System for Identifying Depression Tendencies and Guiding Mindfulness," which is currently under acceptance.
- Successfully developed a VR system for depression detection and auxiliary treatment, providing new technological means and research directions for mental health intervention.

---


### <mark>Research Project Experience 2: Functional Near-Infrared Spectroscopy Classification Based on Deep Learning</mark>

**Project Duration**: September 2024 - November 2024

**Project Description**:  
Responsible for improving the functional near-infrared spectroscopy (fNIRS) classification model. Adopted the **Test Time Training (TTT)** model architecture based on temporal sequences to optimize the existing **Vision Transformer (ViT)**. By replacing the attention layers with the TTT model architecture, the classification performance of the model on temporal data was significantly enhanced. The improved model achieved **State-of-the-Art (SOTA)** performance on three public datasets.

**Technical Details**:

- Analyzed the limitations of existing fNIRS classification models and proposed an improvement plan.
- Adopted the **Test Time Training (TTT)** model architecture based on temporal sequences to optimize the existing **Vision Transformer (ViT)**, enhancing the model's ability to capture temporal features.
- Replaced the attention layers with the TTT model architecture to optimize the model's dynamic adjustment capabilities and improve classification accuracy.
- Conducted experimental validation on three public datasets, using cross-validation and performance evaluation to demonstrate the superiority of the improved model.
- Participating in writing an academic paper detailing the model improvement methods and experimental results, which has been submitted to the **MICCAI conference** (ccfb under review).

**Achievements**:

- Achieved **State-of-the-Art (SOTA)** performance on three public datasets, significantly improving the accuracy of fNIRS classification.
- Submitted the related research results to the MICCAI conference (ccf-b), providing new technical insights for the field of fNIRS classification.



---

### Research Project Experience 3: Research on Automatic Epilepsy Detection System Based on Electroencephalogram (EEG)

**Project Duration**: March 2025 - Present

**Project Description**:  
Dedicated to developing an automatic epilepsy detection system based on electroencephalogram (EEG) signals, utilizing deep learning technology to achieve real-time detection and early warning of epileptic seizures. The project aims to enhance the efficiency and accuracy of epilepsy seizure detection, providing timely medical intervention for patients with epilepsy.

**Technical Details**:

- **EEG Signal Preprocessing**:
  - Studied the fundamental principles of EEG signals, including the characteristics of different frequency bands (such as alpha, beta, delta, and theta waves) and their manifestations during epileptic seizures.
  - Employed techniques such as band-pass filtering, independent component analysis (ICA) to remove ocular artifacts, and baseline correction to preprocess EEG data, ensuring the quality of the input data.

- **Feature Extraction and Model Development**:
  - Utilized **wavelet transform** to extract time-frequency features from EEG signals, capturing local changes and dynamic characteristics within the signals.
  - Designed a model architecture based on **Test Time Training (TTT)**, integrating convolutional neural networks (CNN) and Transformer to enhance the model's ability to recognize epileptic seizures.
  - Developed a hybrid model that combines the strengths of CNN and Transformer to efficiently extract spatiotemporal features, further improving detection accuracy.

- **Real-Time Detection System Design**:
  - Constructed a real-time monitoring system architecture capable of real-time EEG signal acquisition, preprocessing, feature extraction, and detection.
  - Implemented efficient data stream processing algorithms, such as Apache Kafka and TensorFlow Lite, to ensure low-latency system response, enabling timely detection of epileptic seizure signs and issuing early warnings.
  - Designed a multimodal alert mechanism, combining visual and auditory prompts, to ensure that patients and healthcare providers receive early warning information promptly.

- **EEG Analysis System Development**:
  - Created an EEG analysis system capable of analyzing EEG signals over a period of time to automatically identify seizure episodes.
  - Implemented pre-seizure signal detection functionality, which analyzes signals from a period before the seizure to issue early warnings, providing patients with earlier intervention opportunities.

**Achievements**:

- Successfully developed an automatic epilepsy detection system based on deep learning, significantly improving the accuracy and real-time performance of epileptic seizure detection.
- Enhanced the robustness and generalization ability of the system through the optimization of wavelet transform and TTT model architecture.
- The EEG analysis system effectively identifies seizure episodes and issues early warnings, providing an effective technical solution for real-time monitoring and early warning of epilepsy patients.



---

### <mark>Research Project Experience 4: Research on fNIRS-based Brain Signal Decoding</mark>

**Project Duration**: March 2025 - Present

**Project Description**:  
Participated in the research of brain signal decoding based on functional near-infrared spectroscopy (fNIRS), aiming to decode the brain signals of subjects to reconstruct the images they are imagining. The project collects fNIRS signals and language descriptions from subjects while they view images, and uses deep learning technology to achieve multimodal feature fusion and image generation, providing support for understanding the brain's visual information processing mechanisms and developing new brain-computer interface technologies.

**Technical Details**:

- Designed experimental protocols, selecting the COCO dataset as visual stimuli, and synchronously collected fNIRS signals and language descriptions from subjects while they viewed images.
- Preprocessed and extracted features from fNIRS brain signals to identify key features reflecting brain activity.
- Based on the CLIP model architecture, aligned brain signal features to image features, CLIP structural features, and semantic feature spaces to achieve multimodal feature fusion.
- Used deep learning techniques such as Variational Autoencoders (VAE) for image generation tasks, attempting to reconstruct the images imagined by subjects and optimize the generation results.
- Analyzed the similarity between generated and original images to evaluate the decoding effect, and adjusted model parameters and architecture according to experimental results to continuously improve decoding accuracy and image generation quality.

**Achievements**:

- The project is still ongoing. A fNIRS-based brain signal acquisition and processing platform has been successfully established, achieving preliminary alignment of brain signal features with multimodal features.
- Future work will continue to optimize the model to improve decoding accuracy and strive for breakthroughs in the field of brain signal decoding.

---


## Competition Experience: Mathematical Modeling Competitions

**Competition Timeline**:

- Mathematical Contest in Modeling / Interdisciplinary Contest in Modeling (MCM/ICM): February 2024  
- Greater Bay Area Financial Mathematics Modeling Competition: November 2024  
- Asia-Pacific Mathematical Modeling in Business Competition: November 2024  

**Competition Description**:  
In the aforementioned mathematical modeling competitions, I was responsible for preprocessing the raw data, including data cleaning and feature engineering. I utilized methods such as **PCA** and **t-SNE** for dimensionality reduction. Through correlation analysis methods like **MIC** and **Pearson**, I delved into the relationships between data features. I employed various predictive models, including **Random Forest Regression**, **Support Vector Regression (SVR)**, **Neural Network Regression**, **Long Short-Term Memory (LSTM)**, **AutoRegressive Integrated Moving Average (ARIMA)**, and **Grey Prediction**, to model and forecast the data. Ultimately, I submitted high-quality solutions and achieved outstanding results.

**Achievements**:

- Earned the **M Award** in the Mathematical Contest in Modeling / Interdisciplinary Contest in Modeling (MCM/ICM).  
- Received the **Second Prize** in the Greater Bay Area Financial Mathematics Modeling Competition.  
- Won the **Third Prize** in the Asia-Pacific Mathematical Modeling in Business Competition.


## Course Project Experience

### Course Project Experience 1: Multi-functional Guide Device

**Project Duration**: September 2022 - March 2023

**Project Description**:  
Developed a portable multi-functional guide device based on the Raspberry Pi 4B, featuring obstacle avoidance, traffic light recognition and announcement, light sensitivity detection, GPS positioning, and emergency information transmission (location via SMS). The traffic light recognition module utilized the **YOLOv5 model** for high-precision real-time recognition, while other functional modules were connected to the Raspberry Pi via GPIO pins to ensure stable system operation. Through this project, a comprehensive and reliable guide device was successfully implemented, providing all-around support for visually impaired individuals.



---

### Course Project Experience 2: Stock Analysis and Prediction Model Based on Machine Learning

**Project Duration**: November 2023 - January 2024

**Project Description**:  
Responsible for developing a stock analysis and prediction model based on machine learning, primarily utilizing **LSTM (Long Short-Term Memory networks)** to analyze and predict the daily and 5-minute level data of the Hang Seng Index. The project encompassed data preprocessing, factor mining, data splitting, model training, future stock trend prediction, simulated trading, and result visualization. By optimizing trading strategies, an index enhancement strategy was successfully implemented, achieving excess returns under the assumption of an initial investment of 1,000,000 yuan from November 17, 2020, to November 17, 2023.

**Technical Details**:

- Preprocessed the daily and 5-minute level data of the Hang Seng Index, mining relevant factors such as technical indicators and macroeconomic data.
- Used the LSTM model to analyze daily data and predict future stock trends, supporting prediction periods at daily, weekly, and monthly levels.
- Constructed an intraday trading model based on 5-minute level data to identify optimal buy and sell signals, combining these with daily prediction results to formulate trading strategies.
- Simulated the investment process, strictly adhering to real market trading rules, to implement the index enhancement strategy.
- Visualized the final investment results to evaluate the effectiveness of the strategy and model.
- Wrote a detailed course report, clearly explaining the algorithms used, model principles, training process, feature construction methods, and the pros and cons of the strategy.

**Achievements**:

- Successfully developed an LSTM-based stock prediction model, achieving good predictive performance for the Hang Seng Index.
- Implemented index enhancement through optimized trading strategies, obtaining excess returns while keeping maximum drawdown within a reasonable range.
- Completed a detailed course report, clearly demonstrating the entire process of model construction, strategy implementation, and effectiveness evaluation.



---

### <mark>Course Project Experience 3: Independent Sign Language Recognition and Classification Based on Deep Learning</mark>

**Project Duration**: April 2024 - June 2024

**Project Description**:  
Responsible for developing an independent sign language recognition and classification system based on deep learning, handling sequential sign language data with 250 categories. The project included keypoint extraction, feature engineering (adding mean, standard deviation, first-order and second-order differences as features), and designing a model architecture based on **Vision Transformer**. By optimizing model performance through adversarial training, the system achieved an accuracy rate close to **80%** in the 250-category classification task.

**Technical Details**:

- Preprocessed sequential sign language data to extract key-point information, including important joints of the hands and body.
- Conducted feature engineering by adding mean, standard deviation, first-order and second-order differences as features to enhance the model's perception of temporal changes.
- Designed and implemented a deep learning architecture based on **Vision Transformer**, leveraging its powerful feature extraction capabilities to handle complex sequential data.
- Employed adversarial training to enhance the robustness and generalization ability of the model, effectively improving classification accuracy.

**Achievements**:

- Achieved an accuracy rate of **76.8%** in the 250-category sign language classification task, significantly outperforming traditional methods.
- Successfully developed an efficient sign language recognition system, providing technical support for sign language translation and barrier-free communication.



---

### Course Project Experience 4: Fine-tuning of Mental Health Large Language Model Based on LLaMA3

**Project Duration**: September 2024 - November 2024

**Project Description**:  
Responsible for the fine-tuning project of a large language model in the field of mental health, utilizing the **LLaMA3-8B** model. The project involved the collection and organization of mental health datasets, such as counseling dialogues. Through data preprocessing and formatting, the LLaMA3-8B model was fine-tuned to adapt to mental health counseling scenarios. Subsequently, the model was quantized using **llama.cpp** and deployed locally via **LM-Studio**, ensuring the model's efficiency and scalability in practical applications.

**Technical Details**:

- Collected and organized mental health datasets, including counseling dialogue data, to ensure diversity and representativeness.
- Preprocessed and formatted the data, including text cleaning, tokenization, and noise reduction, to prepare for model fine-tuning.
- Fine-tuned the **LLaMA3-8B** model to optimize its performance in mental health counseling, enhancing the accuracy and relevance of dialogue generation.
- Quantized the fine-tuned model using **llama.cpp** to reduce model size and increase inference speed while maintaining performance.
- Deployed the model locally using **LM-Studio** to ensure efficient operation in a local environment, supporting practical applications.

**Achievements**:

- Successfully completed the fine-tuning of the LLaMA3-8B mental health large language model, significantly improving its performance in mental health counseling scenarios.
- Ensured the model's efficiency and scalability in practical applications through quantization and local deployment, providing technical support for intelligent applications in the mental health field.





---

### <mark>Course Project Experience 5: Predicting Binding Affinity of Small Molecules to Specific Protein Targets Using Deep Learning</mark>

**Project Duration**: September 2024 - December 2024

**Project Description**:  
Participated in the development of machine learning models to predict the binding affinity of **small molecules** to **specific protein targets**, a crucial step in drug development. Utilized the **BELKA dataset** (Big Encoded Library for Chemical Assessment) released in April 2024, which contains **thirty million** small molecules with diverse chemical formulas, aiming to predict their binding possibilities with **three specific proteins**.

**Technical Details**:

- Extracted features from small molecules in the BELKA dataset, including **physicochemical properties** and **One-Hot encoded** chemical formula features.
- Conducted data cleaning and feature engineering using **Python** and tools such as **Pandas** and **NumPy** to ensure data quality.
- Applied various machine learning models (e.g., **Random Forest**, **Support Vector Machine**) for preliminary predictions and assessed their performance.
- Designed and implemented deep learning models, including **Convolutional Neural Networks (CNN)** and **Transformer architectures**, to capture complex interactions between small molecules and protein targets.
- Introduced **Graph Neural Networks (GNN)** to leverage the graph-structured features of small molecules, further enhancing prediction accuracy.
- Trained and validated models using **TensorFlow** and **PyTorch** frameworks, optimizing model performance through **cross-validation** and **hyperparameter tuning** to ensure high accuracy and generalization ability.
- Analyzed model prediction results, evaluated the impact of different feature encoding methods on performance, and wrote a project report summarizing the development process, experimental results, and key findings.

**Achievements**:

- Successfully developed deep learning-based prediction models that efficiently predict the binding affinity of **small molecules** to **specific protein targets**.
- Significantly improved prediction accuracy through model comparison and optimization, providing robust technical support for **drug discovery**.
- Provided important theoretical foundations and practical guidance for subsequent **drug development** efforts.