# 2024_2025 SML vs CL

Optional project of the [Streaming Data Analytics](http://emanueledellavalle.org/teaching/streaming-data-analytics-2023-24/](https://emanueledellavalle.org/teaching/streaming-data-analytics-2024-25/) course provided by [Politecnico di Milano](https://www11.ceda.polimi.it/schedaincarico/schedaincarico/controller/scheda_pubblica/SchedaPublic.do?&evn_default=evento&c_classe=811164&polij_device_category=DESKTOP&__pj0=0&__pj1=d563c55e73c3035baf5b0bab2dda086b).

Student: **[To be assigned]**

**DISCAIMER**: This project requires a neural networks/deep learning background.
_____
# Brief Description
This project compares Streaming Machine Learning (SML) and Continual Learning (CL) models on data streams with **temporal dependencies**, under **virtual and real drift scenarios**. We use a weather dataset to predict air temperature based on various weather features, divided into five concepts representing different weather patterns.

The goal is to evaluate how SML, which focuses on rapid adaptation, and CL, which aims to retain past knowledge, handle concept drift and forgetting. The models are assessed through prequential evaluation and CL metrics, including accuracy and backward transfer.

Virtual drift occurs when the distribution of inputs or class priors changes, but the relationship between inputs and outputs remains the same. This means CL models can continue to retain past knowledge as the new concepts are not contradictory. In contrast, real drift involves a shift in the decision boundary, where new concepts contradict or replace old ones, requiring SML models to focus more on rapid adaptation rather than memory retention.

The project involves modifying existing code to handle temporal sequences and drift scenarios using specific models for temporal dependencies (temporal augmentation for SML and LSTM for CL), with deliverables including modified code and a detailed analysis of the results obtained in both drift scenarios.

______

# Background
When applying Machine Learning to data streams, one of the most critical challenges is **concept drift**, which occurs when the statistical properties of the data change over time, potentially impacting the model's ability to make accurate predictions.  

Concept drift can be classified into two types: **virtual drift** and **real drift**. **Virtual drift** happens when the input distribution $P(X|y)$ or class prior $P(y)$ changes, but the relationship between inputs and outputs $P(y|X)$ remains unchanged. This means that while the data may look different, the decision boundary remains the same.

In contrast, **real drift** occurs when $P(y|X)$ changes, meaning the decision boundary shifts. This happens when new patterns contradict previous ones, requiring the model to discard outdated knowledge.

These differences in drift types lead to distinct learning strategies in **Continual Learning (CL)** and **Streaming Machine Learning (SML)**. **CL assumes virtual drift**, where new concepts introduce novel but non-contradictory problems. One can imagine that there is a general problem split into different subproblems, each defined by a specific concept and represented by a specific input distribution $P(X)$ . The objective is to **retain past knowledge** while integrating new information, ensuring that previously learned concepts remain useful. Forgetting is undesirable in this setting because older concepts remain valid. 

**SML, on the other hand, assumes real drift**, where new concepts may **replace old ones** due to shifts in decision boundaries. One can imagine a single problem that continuously changes over time. Here, rapid adaptation takes priority over memory retention, as past knowledge may no longer be relevant or may even contradict the current task. Avoiding forgetting is impossible since the model cannot jointly solve contradictory problems.  

A crucial challenge associated with data streams is that data can exhibit temporal dependence. This dependence arises when a data point's value is influenced by previous observations. It plays a crucial role in modelling real-world processes in data streams, as current outcomes are often correlated with past data. Formally, temporal dependence exists if $\exists \tau \; P(a_t \mid b_{t-\tau}) \neq P(a_t)$ , meaning that a feature or target at time $t$ is statistically dependent on features or targets from previous time steps.  

# Goals and objectives
This project aims to compare SML methodologies and CL strategies on data streams with temporal dependencies in scenarios involving both virtual and real concept drifts. The study evaluates how adaptation-focused models (SML) and models designed to retain past knowledge (CL) perform in these settings, considering both online predictive performance and the impact of forgetting over time.

# Datasets
The **Weather dataset** used for this task consists of weather-related data points, where the goal is to predict a binary label built on the **air temperature** using on the following features:
- **Wind Speed**: The speed of the wind at a given time.
- **Wind Direction**: The direction of the wind at a given time.
- **Humidity**: The percentage of moisture in the air.
- **Dew Point**: The temperature at which air becomes saturated with moisture and forms dew.

Denoting the **target variable** (air temperature) as $v_t$ , the classification functions are built as follows:

1. **F1+**: Predicts **1** if the current temperature exceeds the previous temperature.
   
   $y(X_t) = 1$ if $v_t > v_{t-1}$

   $y(X_t) = 0$ otherwise

2. **F2+**: Predicts **1** if the current temperature is greater than the median of the previous temperatures.

   $y(X_t) = 1$ if $v_t > \text{Median}(v_{t-k}, ..., v_{t-1})$

   $y(X_t) = 0$ otherwise

3. **F3+**: Predicts **1** if the current temperature is greater than the minimum of the previous temperatures.

   $y(X_t) =  1$ if $v_t > \text{Min}(v_{t-k}, ..., v_{t-1})$

   $y(X_t) = 0$ otherwise

4. **F4+**: Predicts **1** if the current temperature increase is greater than the previous temperature increase.
   
   $y(X_t) = 1$ if $\Delta_t > \Delta_{t-1}$

   $y(X_t) = 0$ otherwise

5. **F5+**: Predicts **1** if the current temperature increase is greater than the median of the previous increases.

   $y(X_t) = 1$ if $\Delta_t > \text{Median}(\Delta_{t-k}, ..., \Delta_{t-1})$

   $y(X_t) = 0$ otherwise

Where:
- $v_t$ is the air temperature at time $t$ .
- $\Delta_t = v_t - v_{t-1}$ is the difference between the current and the previous temperature.

### Data Stream Setup

The dataset is already divided into 5 **concepts**, which represent different periods of time or different weather patterns. Each concept corresponds to a data segment, and the data stream will be constructed by utilizing these 5 concepts.
**Important**: Each concept's last 2k data points must be kept aside as the **test set**. The remaining data points will be used for training.

### Virtual Drift Scenario
In the **virtual drift** scenario, you must **standardize each concept differently**, meaning each feature in a concept will have its own mean and variance specific to that concept. However, the **classification function remains the same across all concepts**. We will use **F3+** for all concepts.

**Steps for creating the Virtual Drift data stream**:
1. For each concept, **standardize the features** such that each feature has a **mean and variance specific to that concept** (the same for all the features in the same concept).
2. Use **F3+** for the classification function across all concepts.
3. **Hold out the last 2k data points** from each concept for testing purposes.
4. Build two dataframes: `weather_virtual_train.csv` and `weather_virtual_test.csv`.

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
4.  Build two dataframes: `weather_real_train.csv` and `weather_real_test.csv`.

# **Methodologies and Models to Apply**  
The provided code implements the experiments shown in the **SML vs CL** lecture of the course. However, it must be adapted to work with the new **data stream with temporal dependence**.  

The entry points for running the experiments are:  
- **`run_sml.py`** for **SML models** using River
- **`run_cl.py`** for **CL models**  using Avalanche

#### **SML Models (ARF & ARF_TA)**  

The following SML will be evaluated using **River**:  

- **Adaptive Random Forest (ARF)**: A standard ensemble method for data streams that adapts to concept drifts.  
- **Adaptive Random Forest with Temporal Augmentation (ARF_TA)**: An extension of ARF that incorporates past labels as additional features. TA enhances the feature space by adding the labels of the **previous `o` data points**, allowing the model to leverage temporal dependencies:
  
   $X_t^{T} = X_t \cup \{y_{t-1}, y_{t-2}, ..., y_{t-o} \}$

  This helps ARF_TA capture patterns that standard SML models might miss.  

The execution script **`run_sml.py`** contains the implementation, requiring only minor modifications to be applied to the new data stream.

#### **CL Models (LSTM + Strategies)**  

For **Continual Learning (CL)**, we use an **LSTM-based model** as the base learner, implemented in **`cl_utils.clstm.py`**. The following CL strategies will be tested, all implemented using **Avalanche**:  

- **Naïve**: Plain LSTM without any specific CL strategy.  
- **Experience Replay (ER)**: Stores and replays past data to mitigate forgetting.  
- **ER + Learning without Forgetting (ER + LwF)**: Combines ER with knowledge distillation to retain past knowledge.  
- **Elastic Weight Consolidation (EWC)**: Regularizes updates to prevent drastic changes in important parameters.  
- **Learning without Forgetting (LwF)**: Uses knowledge distillation to transfer knowledge from old to new tasks.  
- **Average Gradient Episodic Memory (AGEM)**: Controls updates using memory gradients to reduce forgetting.  
- **Maximal Interfered Retrieval (MIR)**: Prioritizes replay samples that would be most affected by updates.  

The execution script **`run_cl.py`** contains the implementation but needs modifications to handle temporal dependencies properly.  

##### **Handling Temporal Sequences in CL Models**  

LSTM networks require sequential inputs. To predict the data point at time **t**, the model must process a **sequence of 11 data points**:  

- The feature vectors of the 10 previous time steps*  
- The current feature vector at time t  

Since the first 10 data points in each concept lack enough history to form full sequences, they will receive either **null or random predictions**.  

By structuring the data in this way, we ensure that the LSTM can effectively capture temporal dependencies while applying CL strategies.

# Evaluation Metrics  

Given the class imbalance, we use **Cohen’s Kappa** and **Balanced Accuracy** to provide a fair assessment of classification performance.  

The provided code already implements these metrics, requiring only minor modifications for the new data streams.

#### Prequential Evaluation  
The evaluation follows a **test-then-train** approach, where each incoming data point is first tested and then used for training on the training data stream.
- Concept performance: Performance is reset after each drift to analyze how models perform on entire concepts.  
- Rolling window with window size of 1000: it emphasizes recent data points.  
- Reset rolling window with window size of 1000: The rolling window is reset after each drift, ensuring a fresh evaluation per concept.  

#### Continual Learning Metrics  
To assess forgetting, we compute on the test sets:  
- Average Accuracy: Measures overall performance across all concepts.  
- A-Metric: Evaluates the stability-plasticity trade-off.  
- Backward Transfer (BWT): Measures how learning new concepts affects past knowledge.  
- These metrics are computed using the test set (last 2000 points per concept).  

# Deliverable

For this project, you are required to modify the provided code to apply the methodologies and models to the new data stream with temporal dependencies, as outlined in the previous sections.

1. **Modify the Code**  
   - Adjust the code associated with SML models to handle the new data stream and incorporate temporal dependencies.
   - Modify the CL strategies to manage the new data stream, ensuring correct handling of temporal dependencies and LSTM sequence construction as specified. Ensure your solution handles the required sliding window mechanism for sequence generation.
   - Make sure to apply the correct evaluation metrics (Prequential evaluation and CL metrics) for assessing the performance in both the virtual and real drift scenarios.

2. **Prepare a Presentation (Notebook or PDF)**  
   - Present the results obtained from the two drift scenarios (Virtual Drift and Real Drift).  
   - Compare the models' ability to adapt to new concepts (Prequential Evaluation) and the ability to avoid forgetting (CL Evaluation).  
   - Discuss how the changes in the drift scenarios affect these metrics and the overall performance of the models. Provide insights into the stability-plasticity trade-off for CL models and the adaptability of SML models to new concepts over time.
   - Highlight the impact of temporal dependence on the learning process and model performance in both drift scenarios.

The analysis should be comprehensive and include visualizations, comparisons, and insights into how each model handles the challenges posed by the evolving data stream. Use the plots and the table shown during the SML vs CL lecture.

## Note for Students

* Clone the created repository offline;
* Add your name and surname into the Readme file;
* Make any changes to your repository, according to the specific assignment;
* Add a `requirement.txt` file for code reproducibility and instructions on how to replicate the results;
* Commit your changes to your local repository;
* Push your changes to your online repository.
](https://github.com/Streaming-Data-Analytics/XXXX-XXXX_Project-Title.git)
