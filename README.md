# ARX Model for Time Series Forecasting

This project implements an **ARX (AutoRegressive with eXogenous inputs)** model for forecasting dynamic systems using Python. The ARX model is trained on synthetic time series data, with input and output signals, and is used to forecast unseen output values recursively.

## Project Structure


## How to Run

###  Basic execution:
```bash
python arx_model.py
```

###  With plotting enabled:

```bash
python arx_model.py --plot
```

---

###  With custom paths:

```bash
python arx_model.py \
  --train_input path/to/u_train.npy \
  --train_output path/to/output_train.npy \
  --test_input path/to/u_test.npy \
  --output_file last_400_samples.npy \
  --plot
```

---

## Model Overview

The ARX model predicts the current output `y(k)` using:

- `n` previous output values `y(k-1), ..., y(k-n)`
- `m+1` previous input values `u(k-d), ..., u(k-d-m)`
- A fixed delay `d`

The model equation:

```math
y(k) = φ(k)^T · θ + e(k)
```

Where:

- `φ(k)` is the regressor vector
- `θ` is the coefficient vector learned using linear regression

---

## Features and Visualizations

- Grid search over `(n, d, m)`  
- Model comparison: Least Squares, LassoCV, Ridge  
- Recursive forecasting over test input  
- Heatmap of R² scores (for best `n`)  
- Learning curve (R² vs. train size)  
- Forecast vs. Ground Truth plot

---

## Output

- `last_400_samples.npy`: The final predictions for the last 400 time steps of the test set, required for submission.
- `model_params.json`: Stores the best `(n, d, m)` model parameters found during grid search.

---

## Example Visualizations

The script generates the following if `--plot` is enabled:

- Forecast vs. Actual output time series  
- Heatmap of R² scores for different values of `(d, m)`  
- Learning curve (how R² improves with more training data)

---

## Requirements

- Python 3.7+
- numpy
- scikit-learn
- matplotlib
- seaborn

### Install dependencies:

```bash
pip install -r requirements.txt
```

---

## References

This project is part of a Machine Learning lab course on **System Identification and Regression**.  
The ARX model is commonly used in:

- Industrial control systems  
- Signal processing  
- Econometrics  
- Energy forecasting  

---

## Contact

For questions, reach out to the authors or course staff.  
Enjoy experimenting with ARX modeling! 🚀

