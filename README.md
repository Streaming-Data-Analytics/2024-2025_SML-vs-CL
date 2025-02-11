# XXXX-XXXX Project-Title

Optional project of the [Streaming Data Analytics](http://emanueledellavalle.org/teaching/streaming-data-analytics-2023-24/) course provided by [Politecnico di Milano](https://www11.ceda.polimi.it/schedaincarico/schedaincarico/controller/scheda_pubblica/SchedaPublic.do?&evn_default=evento&c_classe=811164&polij_device_category=DESKTOP&__pj0=0&__pj1=d563c55e73c3035baf5b0bab2dda086b).

Student: **[To be assigned]**


## README structure

# Background
In data stream learning, models must handle continuously evolving data without accessing the entire dataset at once. One of the most critical challenges in this setting is **concept drift**, which occurs when the statistical properties of the data change over time, impacting the model's ability to make accurate predictions.  

Concept drift can be classified into two types: **virtual drift** and **real drift**. **Virtual drift** happens when the input distribution \( P(X|y) \) or class prior \( P(y) \) changes, but the relationship between inputs and outputs \( P(y|X) \) remains unchanged. This means that while the data may look different, the decision boundary remains the same. For example, in weather prediction, shifts in average temperature and atmospheric pressure over time represent virtual drift, as they do not necessarily change the underlying classification rule for extreme weather events.  

In contrast, **real drift** occurs when \( P(y|X) \) changes, meaning the decision boundary itself shifts. This happens when new patterns contradict previous ones, requiring the model to discard outdated knowledge. Using the weather example, real drift might occur due to climate change, where extreme weather events now occur under conditions that previously indicated stability. This alters the classification rule and forces the model to adapt to new patterns rather than relying on past knowledge.  

These differences in drift types lead to distinct learning strategies in **Continual Learning (CL)** and **Streaming Machine Learning (SML)**. **CL assumes virtual drift**, where new concepts introduce novel but non-contradictory problems. The objective is to **retain past knowledge** while integrating new information, ensuring that previously learned concepts remain useful. Forgetting is undesirable in this setting because older concepts remain valid.  

**SML, on the other hand, assumes real drift**, where new concepts **replace old ones** due to shifts in decision boundaries. Here, rapid adaptation takes priority over memory retention, as past knowledge may no longer be relevant or may even contradict the current task. Forgetting is not only necessary but beneficial, as the model must focus exclusively on the present concept to maintain high performance.  

This project aims to compare **SML and CL in data streams with temporal dependencies** under both **virtual and real concept drift**. By analyzing their performance in these different scenarios, we will explore the trade-offs between knowledge retention and fast adaptation, highlighting the strengths and limitations of each approach in dynamic environments.

* 
* Goals and objectives
* Datasets
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
