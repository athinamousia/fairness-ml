Bias mitigation methods are designed to detect and reduce these issues so that the model treats protected groups fairly.

There are three main categories:

| Category            | Purpose                                  | Example techniques                         |
| ------------------- | ---------------------------------------- | ------------------------------------------ |
| **Pre-processing**  | Fix bias in the data before training     | Balancing datasets, Feature transformation |
| **In-processing**   | Add fairness constraints during training | Fairness-aware loss functions              |
| **Post-processing** | Adjust model outputs after training      | Calibrating predictions across groups      |


This section explains the chosen bias mitigation algorithm, presents a practical example, evaluates its performance using fairness metrics and compares results with the original (non-mitigated) model.

