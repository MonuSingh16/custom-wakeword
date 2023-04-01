<h1>  Custome Wake Words Project. </h1> 

### Objective 

-  Voice interaction or commands “Hey Google”, or “Hey Siri” rely on keyword spotting to start interaction on local device. It helps people experience “Hands-free” searching and task completion

### What & Why - Challenges in above technology

- Triggers on negative wake words, unrelated speech, background noise, or silence
- High no. of instance, when device does not trigger on positive wake words 
- Need for quick response & acknowledgment 
- Ability to customise wake word 
- Wake model to be lightweight & energy efficient

### Approach 

- To design, build and deploy a lightweight Keyword spotting ML model (CNN, SVM) and exposed as a mobile-application that can process a “custom wake word”. Voice response with results by respecting local device resource constraints (low compute) and adhering to ethical challenges (Privacy respecting and non-eavesdropping)

### CNN Model Architecture
![alt text](https://github.com/MonuSingh16/custom-wakeword/blob/main/spotify-custom-wakeword/imgs/cnn-model.png?raw=true)

![alt text](https://github.com/MonuSingh16/custom-wakeword/blob/main/spotify-custom-wakeword/imgs/model-summary.png?raw=true)

| Model Profile                    | Value         | Unit    |
|----------------------------------|---------------|---------|
| Selected GPUs                    | None Detected | GPU IDs |
| No. of FLOPs                     | 0.0001        | BFLOPs  |
| GPU Memory Requirement           | 0.1083        | GB      |
| Model Parameters                 | 1.809         | Million |
| Memory Required by Model Weights | 6.9009        | MB      |


### Getting Started

### Local Model deployment Steps:
- Clone the repository into a local machine and enter the `src` directory using
```shell
git clone https://github.com/MonuSingh16/custom-wakeword.git
cd custom-wakeword/spotify-custom-wakeword
```

- Install all the required modules from 'requirements.txt' file using following command
```shell
pip install -r requirements.txt
```

- After all the modules are installed properly, run the following streamlit command
```shell
streamlit run app/main.py
```

- After running above command a browser should open, which contains the Wakeword Detection application

### Streamlit App
![alt text](https://github.com/MonuSingh16/custom-wakeword/blob/main/spotify-custom-wakeword/imgs/streamlit-app-demo.png?raw=true)

### Cloud Model Deployment Architecture 
![alt text](https://github.com/MonuSingh16/custom-wakeword/blob/main/spotify-custom-wakeword/imgs/architecture.png?raw=true)




### Powerpoint
- [Presentation slides](https://docs.google.com/presentation/d/1FsmwdQwOyUFbaz1h2P893Sn3dWo6Cn_E/edit?usp=sharing&ouid=100632006093746569721&rtpof=true&sd=true)

### Team Members
 - Monu Singh, Nischal, Nkem Michael Onuorah

