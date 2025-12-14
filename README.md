# UFC Fight Prediction Model

This project trains and applies a machine learning model to predict win probabilities for UFC fights based on fighter history, rankings, performance statistics, and physical attributes (such as age, height, and reach).

The workflow separates model training and inference, allowing the trained model to be reused to score new or upcoming fights without retraining.

The data used in this project is sourced from the [Ultimate UFC Dataset](https://github.com/shortlikeafox/ultimate_ufc_dataset) repository by Matthew Dabbert. You can find more information about this dataset on [Kaggle](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset).

When new `upcoming.csv` files are added, the trained model can be applied to these datasets using `apply_model.ipynb` to generate predicted win probabilities for upcoming fights.
