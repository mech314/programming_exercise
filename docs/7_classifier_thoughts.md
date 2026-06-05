### Thoughts

- Since we are we have much more genes then samples ~4000 vs 250
  decision tress will probably overfit. 
- It is counter intuitive, but I would start with L2 logistic regression. 


### Evaluation

Model's predictive capacity looks decent: 

- Balanced accuracy 0.89, 
- F1 0.89
- ROC AUC 0.985. 

Predictive performance is good across all four subtypes (F1 0.86-0.92), 
balancing classes definitely helped with unbalanced (Proliferative n=40) class