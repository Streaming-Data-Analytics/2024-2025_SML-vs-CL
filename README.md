# 2024_2025 SML vs CL

Optional project of the [Streaming Data Analytics](http://emanueledellavalle.org/teaching/streaming-data-analytics-2023-24/) course provided by [Politecnico di Milano](https://www11.ceda.polimi.it/schedaincarico/schedaincarico/controller/scheda_pubblica/SchedaPublic.do?&evn_default=evento&c_classe=811164&polij_device_category=DESKTOP&__pj0=0&__pj1=d563c55e73c3035baf5b0bab2dda086b).

Student: **[To be assigned]**

# Background
When applying Machine Learning to data streams, one of the most critical challenges in this setting is **concept drift**, which occurs when the statistical properties of the data change over time, impacting the model's ability to make accurate predictions.  

Concept drift can be classified into two types: **virtual drift** and **real drift**. **Virtual drift** happens when the input distribution $ P(X|y) $ or class prior $ P(y) $ changes, but the relationship between inputs and outputs $ P(y|X) $ remains unchanged. This means that while the data may look different, the decision boundary remains the same. For example, shifts in average temperature and atmospheric pressure over time in weather prediction represent virtual drift, as they do not necessarily change the underlying classification rule for extreme weather events.  

In contrast, **real drift** occurs when $ P(y|X) $ changes, meaning the decision boundary shifts. This happens when new patterns contradict previous ones, requiring the model to discard outdated knowledge. Using the weather example, real drift might occur due to climate change, where extreme weather events now occur under previously indicated stability conditions. This alters the classification rule and forces the model to adapt to new patterns rather than relying on past knowledge.  

These differences in drift types lead to distinct learning strategies in **Continual Learning (CL)** and **Streaming Machine Learning (SML)**. **CL assumes virtual drift**, where new concepts introduce novel but non-contradictory problems. One can imagine that there is a general problem split into different subproblems, each defined by a specific concept and represented by a specific input distribution $P(X)$. The objective is to **retain past knowledge** while integrating new information, ensuring that previously learned concepts remain useful. Forgetting is undesirable in this setting because older concepts remain valid. 

**SML, on the other hand, assumes real drift**, where new concepts may **replace old ones** due to shifts in decision boundaries. One can imagine a single problem that continuously changes over time. Here, rapid adaptation takes priority over memory retention, as past knowledge may no longer be relevant or may even contradict the current task. Avoiding forgetting is impossible since the model cannot jointly solve contradictory problems.  

A crucial challenge associated with data streams is that data can exhibit temporal dependence. This dependence arises when a data point's value is influenced by previous observations. It plays a crucial role in modelling real-world processes in data streams, as current outcomes are often correlated with past data. Formally, temporal dependence exists if $\exists \tau \; P(a_t \mid b_{t-\tau}) \neq P(a_t)$, meaning that a feature or target at time $ t $ is statistically dependent on features or targets from previous time steps.  
Temporal dependence appears in many domains: IoT sensors track environmental patterns, video surveillance relies on past frames for object tracking, satellite imagery captures spatial continuity over time, and robotics decisions depend on previous states.  

* Goals and objectives
This project aims to compare SML methodologies and CL strategies on data streams with temporal dependencies in scenarios involving both virtual and real concept drifts. The study evaluates how adaptation-focused models (SML) and models designed to retain past knowledge (CL) perform in these settings, considering both online predictive performance and the impact of forgetting over time.

# Datasets
The **Weather dataset** used for this task consists of weather-related data points, where the goal is to predict the **air temperature** based on various features. The dataset includes multiple features collected from weather stations, such as:
- **Wind Speed**: The speed of the wind at a given time.
- **Wind Direction**: The speed of the wind at a given time.
- **Humidity**: The percentage of moisture in the air.
- **Dew Point**: The temperature at which air becomes saturated with moisture and forms dew.

The **target variable** for classification is the **air temperature**, denoted as $ v_t $, a continuous value we want to predict using the other features in the dataset. The classification task is binary: we need to predict whether the current air temperature is higher or lower than a certain threshold, and this prediction is made using the other environmental features.

### Classification Functions
Here are the **classification functions** used to determine the labels based on the temperature values:

1. **F1+**: Predicts **1** if the current temperature exceeds the previous temperature.
   $y(X_t) = \begin{cases} 1, & \text{if } v_t > v_{t-1} \\0, & \text{otherwise} \end{cases}$

2. **F2+**: Predicts **1** if the current temperature is greater than the median of the previous temperatures.
   $
   y(X_t) = 
   \begin{cases} 
   1, & \text{if } v_t > \text{Median}(v_{t-k}, ..., v_{t-1}) \\
   0, & \text{otherwise}
   \end{cases}
   $

3. **F3+**: Predicts **1** if the current temperature is greater than the minimum of the previous temperatures.
   $
   y(X_t) = 
   \begin{cases} 
   1, & \text{if } v_t > \text{Min}(v_{t-k}, ..., v_{t-1}) \\
   0, & \text{otherwise}
   \end{cases}
   $

4. **F4+**: Predicts **1** if the current temperature increase is greater than the previous temperature increase.
   $
   y(X_t) = 
   \begin{cases} 
   1, & \text{if } \Delta_t > \Delta_{t-1} \\
   0, & \text{otherwise}
   \end{cases}
   $

5. **F5+**: Predicts **1** if the current temperature increase is greater than the median of the previous increases.
   $
   y(X_t) = 
   \begin{cases} 
   1, & \text{if } \Delta_t > \text{Median}(\Delta_{t-k}, ..., \Delta_{t-1}) \\
   0, & \text{otherwise}
   \end{cases}
   $



Where:
- $ v_t $ is the air temperature at time $ t $.
- $ \Delta_t = v_t - v_{t-1} $ is the difference between the current and the previous temperature.

### Data Stream Setup

The dataset is already divided into 5 **concepts**, which represent different periods of time or different weather patterns. Each concept corresponds to a data segment, and the data stream will be constructed by utilizing these 5 concepts.
**Important**: Each concept's last 2k data points must be kept aside as the **test set**. The remaining data points will be used for training.

### Virtual Drift Scenario
In the **virtual drift** scenario, you must **standardized each concept differently**, meaning each feature in a concept will have its own mean and variance specific to that concept. However, the **classification function remains the same across all concepts**. We will use **F3+** for all concepts, which means the model will predict **1** if the current temperature exceeds the minimum of the previous temperatures.

**Steps for creating the Virtual Drift data stream**:
1. For each concept, **standardize the features** such that each feature has a **mean and variance specific to that concept** (the same for all the features in the same concept).
2. Use **F3+** (based on the minimum of previous temperatures) for the classification function across all concepts.
3. **Hold out the last 2k data points** from each concept for testing purposes.

### Real Drift Scenario
In the **real drift** scenario, each concept is **standardized in the same way**, meaning the features will have **mean 0** and **variance 1** for each concept. However, in contrast to the virtual drift scenario, the **classification function will change for each concept**.

**Steps for creating the Real Drift data stream**:
1. For each concept, **standardize the features** so that each feature has a **mean of 0** and **variance of 1** for that concept.
2. Use the appropriate classification function for each concept:
   - **Concept 1**: Use **F1+**.
   - **Concept 2**: Use **F2-**.
   - **Concept 3**: Use **F3+**.
   - **Concept 4**: Use **F4-**.
   - **Concept 5**: Use **F5+**.
3. **Hold out the last 2k data points** for each concept to be used as the test set.

* Methodologies/models to apply
* Evaluation metrics
* Deliverable 

## Note for Students

* Clone the created repository offline;
* Add your name and surname into the Readme file;
* Make any changes to your repository, according to the specific assignment;
* Add a `requirement.txt` file for code reproducibility and instructions on how to replicate the results;
* Commit your changes to your local repository;
* Push your changes to your online repository.

5. **F5+**: Predicts **1** if the current temperature increase is greater than the median of the previous increases.
   \[
   y(X_t) = 
   \begin{cases} 
   1, & \text{if } \Delta_t > \text{Median}(\Delta_{t-k}, ..., \Delta_{t-1}) \\
   0, & \text{otherwise}
   \end{cases}
   \]
