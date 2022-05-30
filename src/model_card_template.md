# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project utilizes Random Forest Classifier machine learning alogrithm from Sklearn to predict a persons salary. The prediction is based on $50,000 thershold. The model will predict if someone's salary is above $50,000 based on certain features as inputs to the model. 

## Intended Use
The model is used to predict the salary of a person based on data publicly available from the Census Bureau. 

## Training Data
Census Bureau data was used in the training of this model. The data contains features like age, salary information, occupation. More information on the data can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income)

## Evaluation Data
Using the same data from the Census Bureau, 20% of it was used for evaluation (note that the evaulation data was not used for training the model).

## Metrics
Below are the metrics pulled from the final model:  
- `Precision: 0.7486792452830189`  
- `Recall: 0.6215538847117794`  
- `F-beta: 0.6792194453954126`  


## Ethical Considerations
Looking at the training data, some information should not be used to determine the salary information (e.g. race, sex). These factors can be biased and cause discrepancy if the model results were utilized to make any decisions.   
Further more, including multiple countries with different cost of living can effect the model results and biasing towards countries with higher cost of living. 

## Caveats and Recommendations
