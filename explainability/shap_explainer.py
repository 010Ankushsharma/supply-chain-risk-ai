import shap
import matplotlib.pyplot as plt
import os

OUTPUT_PATH = "models/artifacts/"

def generate_shap(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)

    filepath = os.path.join(OUTPUT_PATH, "shap_summary.png")
    plt.savefig(filepath)

    print("SHAP plot saved")