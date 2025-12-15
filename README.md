# amfam

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/laurenkhoury/amfam" style="text-decoration: none !important;"><img src="images/uw_logo.png" alt="logo1" width="200" style="vertical-align: bottom;"><img src="images/amfam_logo.png" alt="logo2" width="200" style="vertical-align: bottom;"></a>
</div>

## About the Project

Welcome to our Capstone project for our Master's program in Data Science in Human Behavior (DSHB)!
In this project, under the guidance of collaborators at American Family Insurance, we aim to study human migration patterns from hurricane-prone areas by employing the concept of digital twins. We collect data from multiple public sources to create personas which are then used to build generative agents. We run machine learning models with ground-truth migration data to extract feature importances. We interact with the agents via an LLM and run prompt engineering to refine the prompt sent to the LLM to ensure consistency in responses. We follow the generative agent framework detailed in this repository: https://github.com/StanfordHCI/genagents   

## Files

- `archive`
    - contains all files from previous cohort's repository for this project which can also be found here [DSHB-Capstone-AMFAM](https://github.com/lsmithbecker/DSHB-Capstone-AMFAM)
 
- `code`
    - `ML_models`
        - contains code used to run machine learning models (GLMER, Random Forest, XGBoost) 
    - `personas`
        -  contains code to create personas and format each in a JSON file
    - `prompts`
        - contains code used for prompt engineering and 3 versions of the engineered prompt output (Academic, Deterministic, Social)
    - `gen_agents`
        - contains code for building the generative agents and interacting with the agents
     
- `data`
  - contains public datasets used to build personas along with a data dictionary
  - contains the final persona file with 80 rows of example personas
 
## Authors

- David Jolly
- Lauren Khoury
- Jake Murray
- Andy Vo
- Ho Wong
- Dusk Zhang
