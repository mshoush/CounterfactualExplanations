# Exploring explainable AI through the use of counterfactual explanations


This project provides supplementary material for the Medium article ["Exploring explainable AI through the use of counterfactual explanations"]([https://arxiv.org/abs/2307.06564](https://medium.com/@hrm.michael/exploring-explainable-ai-through-the-use-of-counterfactual-explanations-d66c2c08e8ec)) by Mahmoud Shoush, Mihhail Sokolov, Susanna Metsla, and Teele Kuri. It is associated with the neural network course at the ["Institute of Computer Science at the University of Tartu."](https://courses.cs.ut.ee/2024/nn/spring) 


We replicate the results of ["Explaining the black-box smoothly — A counterfactual approach” by Singla et al. (2023)"](https://www.sciencedirect.com/science/article/abs/pii/S1361841522003498?via=ihub) and ["COIN: Counterfactual inpainting for weakly supervised semantic segmentation for medical images” by Shvetsov et al. (2024)"](https://arxiv.org/abs/2404.12832) in producing explainable counterfactuals for a black-box classification model for medical imaging. Also, we acknowledge that most of the code here is taken from ["Explainable AI Using Generative Adversarial Networks "](https://github.com/Dmytro-Shvetsov/counterfactual-search/tree/main).


# Dataset: 
* [COVID-19 Dataset](https://www.kaggle.com/datasets/imdevskp/corona-virus-report)



# Reproduce results:
To reproduce the results, please run the following:

* First, install the required packages using the following command into your environment: 

                                  pip install --no-cache-dir -r ./counterfactual-search/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
                                  pip install -e .
                                  pre-commit install
                  

* Next, download the data folder from the abovementioned link

* Run the following notebooks to replicate the experiments:
  
                                  run_all.ipynb
                         

  
*   Run the following shell script to start experiments w.r.t the offline phase: 

                                     ./run_offline_phase.sh
    
*   Execute the following shell script to initiate experiments with varying resource availability, thereby obtaining resource utilization levels:

                                     ./run_extract_utilization_levels.sh

    
*   Compile results to extract the resource utilization levels by executing the following notebook:

                                     extract_resource_utilization_levels.ipynb


*   Run the following shell script to conduct experiments involving different variants of the proposed approach as well as baseline methods:

                                    ./run_variants_with_BLs.sh <log_name> <resFolder> <mode> <resource_levels>
                                    log_name: ["bpic2012", "bpic2017", "trafficFines"]
                                    mode: ["BL1", "BL2", "ours" ]
                                    resource_levels: as extracted from the previous step.
    
                                    Ex: taskset -c 0-7 ./run_variants_with_BLs.sh bpic2012  resultsRL ours  "1 4 6 12"
 
                                     

* Finally, execute the following notebook to collect results regarding RQ1 and RQ2.: 

                                     compile_results.ipynb
                                     




