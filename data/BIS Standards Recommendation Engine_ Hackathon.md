# **BIS Standards Recommendation Engine: Hackathon**

## **Official Rulebook**

## **1\. EVENT OVERVIEW**

**Theme:** Accelerating MSE Compliance – Automating BIS Standard Discovery  


**Track**: AI / Retrieval Augmented Generation (RAG)

### **1.1 The Challenge**

Indian Micro and Small Enterprises (MSEs) often spend weeks identifying which Bureau of Indian Standards (BIS) regulations apply to their products. We are looking for an AI-powered **Recommendation Engine** that uses RAG to turn product descriptions into accurate standard recommendations in seconds.

### **1.2 Required Solution**

You must build a proof-of-concept that:

1. Accepts a product description.  
2. Recommends the top relevant BIS standards with a brief rationale.  
3. Focuses on depth within the **Building Materials** category (e.g., Cement, Steel, Concrete, Aggregates).

## **2\. PARTICIPATION**


### **2.2 Rules of Engagement**

* **Autonomy**: You can use open-source libraries, frameworks, and pre-trained models.   

## **3\. TECHNICAL REQUIREMENTS**

### **3.1 Deliverables**

1. **Functional RAG Prototype**: An end-to-end pipeline (Retriever → LLM → Output) that provides top 3–5 standards for a given product.  
2. **Source Code**: A public GitHub repository with a clean README and documented code.  
3. **Mandatory Eval Script**: Teams MUST include the Python script eval\_script.py (provided at Hour 0\) in their submission. The script accepts a JSON of query-result pairs and outputs Hit Rate @3, MRR @5, and Latency.  
4. **Presentation & Demo**: A recorded demo no longer than 7 minutes, and an 8-slide deck structured as follows: (1) Problem Statement, (2) Solution Overview, (3) System Architecture, (4) Chunking & Retrieval Strategy, (5) Demo Highlights, (6) Evaluation Results, (7) Impact on MSEs, (8) Team & Acknowledgements.

### **3.2 Constraints**

* **Dataset Integrity**: All metrics use the provided dataset as the sole source of truth.  
* **Transparency**: Disclose all external APIs and data sources used.  
* **Hardware**: The system must be fully runnable using the submitted GitHub repository, running on standard hardware (consumer GPUs are fine).Document all environment dependencies clearly in your README.

### **3.3 Evaluation Script Specification**

Your GitHub repository **MUST** contain a script named inference.py at the root level that accepts an \--input argument and an \--output argument. The judges will evaluate your project by running the following command: python inference.py \--input hidden\_private\_dataset.json \--output team\_results.json. Your inference.py must:

1. Read the JSON file provided in the \--input path.  
2. Pass the query field of each item through your RAG pipeline.  
3. Save the results to the path specified in \--output using the strict JSON format (containing id, retrieved\_standards, and latency\_seconds).

If your code cannot be run via this command, you will score a 0 for the automated metrics.

## **4\. JUDGING & SCORING (100 Points Total)**

### **4.1 Primary Metrics (Automated \- 40 Points)**

***Note: If eval\_script.py crashes due to any reason (i.e. changed key name or any other reason), the team scores a zero for the automated portion. Participants should strictly follow the provided output json schema.***

The following metrics will be calculated using the **Hidden Private Test Set**:

| Metric | Definition | Formula | Target |
| :---- | :---- | :---- | :---- |
| **Hit Rate @3** | % of queries where at least 1 expected standard appears in top-3 results | (correct\_queries / total\_queries) × 100 | \>80% |
| **MRR @5** | Mean Reciprocal Rank of first correct standard in top-5 | Σ(1/rank\_i) / N | \>0.7 |
| **Avg Latency** | Average response time per query | total\_time / num\_queries | \<5 seconds |

### 

### **4.2 Secondary Metrics (Manual \+ Semi-Auto \- 20 Points)**

| Metric | Definition | Scoring | Judge Role |
| :---- | :---- | :---- | :---- |
| **Relevance Score** | Judge rates top-3 results (1-5 scale) | Avg. of 5 random queries per team | Manual review |
| **No Hallucinations** | % responses containing imaginary standards.  | Binary check per response. Scored as binary per response: 1 (clean) or 0 (hallucination detected). Final score \= (clean responses / total responses) × 10\. | Manual \+ Keyword filter |

### 

### **4.3 Subjective Scoring (40 Points)**

* **Technical Excellence (10 pts)**: Code quality, pipeline logic, and reproducibility.  
* **Innovation (10 pts)**: Unique chunking or retrieval strategies.  
* **Usability & Impact (10 pts)**: UI/UX quality and relevance to MSE use cases.  
* **Presentation (10 pts)**: Clarity and storytelling during the demo. The presentation will be conducted online via Google Meet after the submission deadline (i.e. 3rd May 2026, 11:59pm)

## **5\. SUBMISSION GUIDELINES**

### **5.1 Package Structure**

Your GitHub repo should include:

* /src: Main application logic.  
* /data: Results from the public test set.  
* eval\_script.py: The mandatory evaluation script provided by organizers.  
* inference.py: The mandatory entry-point script for judges.  
* requirements.txt: All dependencies specified.  
* presentation.pdf: Your slide deck.

## **6\. DATASET & EVALUATION**

**Official Dataset**: Participants will be provided with the **raw PDF documents** of the **BIS SP 21 (Summaries of Indian Standards for Building Materials)**. Teams are responsible for their own data ingestion, parsing, and chunking strategy.

**Evaluation Tiers**:

* **Public Test Set**: 10 sample queries provided at kickoff for local validation.  
* **Private Test Set**: A hidden set of queries used by judges to calculate automated scores.




