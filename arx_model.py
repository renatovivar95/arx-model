import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import argparse
import os
import json
import seaborn as sns

# ---------- ARX Feature Vector ----------
def phi(k, n, d, m, u, y): 
    phivector = []
    for n_ in range(1, n+1):
        if k - n_ >= 0:
            phivector.append(y[k - n_])
    for m_ in range(m + 1):
        if k - d - m_ >= 0:
            phivector.append(u[k - d - m_])
    return phivector

def build_regressor_matrix(n, d, m, u, y, start, end):
    X_vector, Y_vector = [], []
    p = max(n, d + m)
    for i in range(p, end):
        if i >= start:
            X_vector.append(phi(i, n, d, m, u, y))
            Y_vector.append(y[i])
    return np.array(X_vector), np.array(Y_vector)

def coeff_determination(actual, predicted):
    SS_res = np.sum((actual - predicted) ** 2)
    SS_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - SS_res / SS_tot

def load_npy_file(file_path):
    return np.load(file_path)

def print_return_data_informasjon(Data_X_train, Data_Y_train):
    X_train, X_test, y_train, y_test = train_test_split(Data_X_train, Data_Y_train, test_size=0.33, shuffle=False)
    return X_train, X_test, y_train, y_test

# ---------- Grid Search ----------
def best_n_d_m(U_train, U_test, Output_train, Output_test, plot_heatmap=True):
    best_d = best_n = best_m = 0
    best_r2 = -np.inf
    best_predicted = []
    r2_scores = {}
    model = LinearRegression()
    for n in range(10):
        for d in range(10):
            for m in range(10):
                X_train, Y_train = build_regressor_matrix(n, d, m, U_train, Output_train, 0, len(Output_train))
                X_test, Y_test = build_regressor_matrix(n, d, m, U_test, Output_test, 0, len(Output_test))
                if len(X_test) == 0 or len(X_train) == 0:
                    continue
                model.fit(X_train, Y_train)
                predicted = model.predict(X_test)
                r2 = coeff_determination(Y_test, predicted)
                r2_scores[(n, d, m)] = r2
                if r2 > best_r2:
                    best_r2 = r2
                    best_predicted = predicted
                    best_n, best_d, best_m = n, d, m

    if plot_heatmap:
        plot_r2_heatmap(r2_scores, fixed_n=best_n)

    return best_n, best_d, best_m, best_predicted

# ---------- Recursive Prediction ----------
def append_to_END_Y(Data_U, Data_Output, Data_test, n, d, m):
    p = max(n, d + m)
    U_combined = np.concatenate((Data_U[-p:], Data_test))
    Y = list(Data_Output)
    X, Y_slice = build_regressor_matrix(n, d, m, Data_U, Data_Output, p, len(Data_Output))
    model = LinearRegression().fit(X, Y_slice)
    for i in range(len(Data_test)):
        new_phi = []
        for n_ in range(1, n+1):
            new_phi.append(Y[-n_])
        for m_ in range(m + 1):
            new_phi.append(U_combined[p + i - d - m_])
        y_new = model.predict([new_phi])[0]
        Y.append(y_new)
    return Y

# ---------- Grid Search Heatmap ----------
def plot_r2_heatmap(r2_scores, fixed_n):
    r2_matrix = np.zeros((10, 10))
    for (n, d, m), r2 in r2_scores.items():
        if n == fixed_n:
            r2_matrix[d, m] = r2
    plt.figure(figsize=(8, 6))
    sns.heatmap(r2_matrix, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"R² Heatmap (fixed n={fixed_n})")
    plt.xlabel("m")
    plt.ylabel("d")
    plt.tight_layout()
    plt.savefig("PLOTS/r2_heatmap.png")
    plt.close()

# ---------- Learning Curve ----------
def plot_learning_curve(n, d, m, U_train, Output_train, U_test, Output_test):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_r2 = []
    test_r2 = []
    N = len(Output_train)

    for frac in train_sizes:
        size = int(N * frac)
        X_train, Y_train = build_regressor_matrix(n, d, m, U_train[:size], Output_train[:size], 0, size)
        X_test, Y_test = build_regressor_matrix(n, d, m, U_test, Output_test, 0, len(Output_test))
        if len(X_train) == 0 or len(X_test) == 0:
            train_r2.append(np.nan)
            test_r2.append(np.nan)
            continue
        model = LinearRegression().fit(X_train, Y_train)
        train_r2.append(coeff_determination(Y_train, model.predict(X_train)))
        test_r2.append(coeff_determination(Y_test, model.predict(X_test)))

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_r2, label="Train R²")
    plt.plot(train_sizes, test_r2, label="Test R²")
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("R² Score")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PLOTS/learning_curve.png")
    plt.close()

# ---------- Main Function ----------
def main(args):
    Data_U = load_npy_file(args.train_input)
    Data_Output = load_npy_file(args.train_output)
    Data_test = load_npy_file(args.test_input)
    assert Data_U.ndim == 1 and Data_test.ndim == 1 and Data_Output.ndim == 1
    U_train, U_test, Output_train, Output_test = print_return_data_informasjon(Data_U, Data_Output)

    n, d, m, predicted = best_n_d_m(U_train, U_test, Output_train, Output_test, plot_heatmap=True)
    model_params = {'n': n, 'd': d, 'm': m}
    with open("model_params.json", "w") as f:
        json.dump(model_params, f)
    print(f"Saved model parameters to model_params.json")

    X_train, Y_train = build_regressor_matrix(n, d, m, U_train, Output_train, 0, len(Output_train))
    X_test, Y_test = build_regressor_matrix(n, d, m, U_test, Output_test, 0, len(Output_test))

    model_ls = LinearRegression().fit(X_train, Y_train)
    model_lasso = LassoCV(cv=TimeSeriesSplit(n_splits=5)).fit(X_train, Y_train)
    model_ridge = Ridge(alpha=0.2).fit(X_train, Y_train)

    Y_pred_ls = model_ls.predict(X_test)
    Y_pred_lasso = model_lasso.predict(X_test)
    Y_pred_ridge = model_ridge.predict(X_test)

    print(f"R² (Least Squares): {coeff_determination(Y_test, Y_pred_ls):.4f}")
    print(f"R² (LassoCV): {coeff_determination(Y_test, Y_pred_lasso):.4f}")
    print(f"R² (Ridge α=0.2): {coeff_determination(Y_test, Y_pred_ridge):.4f}")

    full_result = append_to_END_Y(Data_U, Data_Output, Data_test, n, d, m)
    last_400 = np.array(full_result[-400:])
    np.save(args.output_file, last_400)
    print(f"Saved last 400 predictions to {args.output_file}")

    if args.plot:
        # Full output vs input plot
        plt.plot(full_result, linestyle=':', label='Predicted Y')
        plt.plot(Data_Output, label='True Y (Train)')
        plt.plot(np.concatenate((Data_U, Data_test)), label='Input U')
        plt.legend()
        plt.tight_layout()
        plt.savefig("PLOTS/comparison_test_train_data.png")
        plt.close()

        # Test prediction vs true comparison
        plt.figure(figsize=(10, 5))
        plt.plot(Y_test, label='True Y (Test)', linewidth=2)
        plt.plot(Y_pred_ls, '--', label='Predicted Y (LS)')
        plt.plot(Y_pred_lasso, '--', label='Predicted Y (Lasso)')
        plt.plot(Y_pred_ridge, '--', label='Predicted Y (Ridge)')
        plt.xlabel('Time Index')
        plt.ylabel('Output Y')
        plt.title('Comparison on Test Set')
        plt.legend()
        plt.tight_layout()
        plt.savefig("PLOTS/prediction_vs_test.png")
        plt.close()

    plot_learning_curve(n, d, m, U_train, Output_train, U_test, Output_test)

# ---------- Entry Point ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ARX model with visualizations")
    parser.add_argument('--train_input', type=str, default='DATA/u_train.npy')
    parser.add_argument('--train_output', type=str, default='DATA/output_train.npy')
    parser.add_argument('--test_input', type=str, default='DATA/u_test.npy')
    parser.add_argument('--output_file', type=str, default='DATA/last_400_samples.npy')
    parser.add_argument('--plot', action='store_true', help='Show prediction plot')
    args = parser.parse_args()
    main(args)
