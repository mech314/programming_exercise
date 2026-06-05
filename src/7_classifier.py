import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)


def get_args():
    pass


def load_data():
    pass


def build_pipeline():
    pass


def ev_cv():
    pass


def plot_stats():
    pass


def train_final():
    pass


def main():
    pass


if __name__ == "__main__":
    main()