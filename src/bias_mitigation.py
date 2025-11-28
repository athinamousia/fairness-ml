import numpy as np
from sklearn.metrics import make_scorer
from src.utils import (
    read_csv,
    load_from_pickle,
    split_data,
    average_odds_scoring,
    run_grid_search,
    calculate_performance_and_fairness_metrics,
    build_metrics_list,
    plot_metrics_grid,
)
from src.constants import (
    EVAL_METRICS,
    FAIRNESS_METRICS,
    XGB_PARAMS,
    LFR_PARAMS,
    ADVERSARIAL_PARAMS,
    EGR_PARAMS,
    PROTECTED_ATTRS,
)
from aif360.sklearn.preprocessing import Reweighing, LearnedFairRepresentations
from aif360.sklearn.inprocessing import (
    AdversarialDebiasing,
    ExponentiatedGradientReduction,
)
from aif360.sklearn.postprocessing import (
    CalibratedEqualizedOdds,
    RejectOptionClassifierCV,
    PostProcessingMeta,
)
from tensorflow.python.framework.ops import disable_eager_execution


class PreprocessingModel:
    """
    A class that handles:
    - Baseline model
    - Fairness mitigations (Reweighing, LFR)
    - Metrics & plots
    """

    def __init__(self):
        self.input_file = "data/standard_df.csv"
        self.protected_attrs = PROTECTED_ATTRS
        self.target = "Target"

        self.df = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.baseline_df = None

    def load_and_split(self):
        self.df = read_csv(self.input_file)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            *_,
        ) = split_data(self.df, self.protected_attrs, self.target)

    def load_baseline(self):
        self.baseline_df = read_csv("output/csv/baseline_xgboost.csv")
        self.baseline_model = load_from_pickle("output/model/baseline_model.pkl")

    # Reweighing
    def run_reweighing(self):
        scorer = make_scorer(
            average_odds_scoring,
            scoring="statistical_parity",
            greater_is_better=False,
            protected_attrs=self.protected_attrs[0],
        )

        weights = Reweighing(self.protected_attrs).fit_transform(
            self.X_train, self.y_train
        )[1]

        reweigh_grid = run_grid_search(
            self.baseline_model,
            {},
            self.X_train,
            self.y_train,
            scorer,
            weights,
        )
        y_pred = reweigh_grid.predict(self.X_test)
        y_proba = reweigh_grid.predict_proba(self.X_test)[:, 1]

        reweigh_df = calculate_performance_and_fairness_metrics(
            self.y_test, y_pred, y_proba, self.protected_attrs
        )
        reweigh_df.to_csv("output/csv/reweighing.csv", index=False)

        labels = ["Evaluation Metrics"] + [
            f"{prot_attr} Fairness Metrics" for prot_attr in self.protected_attrs
        ]

        metrics_list = build_metrics_list(
            reweigh_df,
            self.baseline_df,
            EVAL_METRICS,
            FAIRNESS_METRICS,
            "Reweighing",
            self.protected_attrs,
        )
        plot_metrics_grid(metrics_list, labels, ["Reweighing", "XGBoost"])

    # Learned Fair Representation
    def run_lfr(self):
        lfr = LearnedFairRepresentations(
            self.protected_attrs[0],
            n_prototypes=25,
            max_iter=1000,
            random_state=0,
        )

        scorer_lfr = make_scorer(
            average_odds_scoring,
            scoring="delta",
            protected_attrs=self.protected_attrs[0],
        )

        lfr_grid = run_grid_search(
            lfr, LFR_PARAMS, self.X_train, self.y_train, scorer_lfr
        )

        y_pred_lfr = lfr_grid.predict(self.X_test)
        y_proba_lfr = lfr_grid.predict_proba(self.X_test)[:, 1]

        lfr_df = calculate_performance_and_fairness_metrics(
            self.y_test, y_pred_lfr, y_proba_lfr, self.protected_attrs
        )
        lfr_df.to_csv("output/csv/lfr.csv", index=False)

        labels = ["Evaluation Metrics"] + [
            f"{prot_attr} Fairness Metrics" for prot_attr in self.protected_attrs
        ]
        metrics_list_lfr = build_metrics_list(
            lfr_df,
            self.baseline_df,
            EVAL_METRICS,
            FAIRNESS_METRICS,
            "LFR",
            self.protected_attrs,
        )
        plot_metrics_grid(metrics_list_lfr, labels, ["LFR", "XGBoost"])

    # Run all steps
    def run_preprocessing_models(self):
        self.load_and_split()
        self.load_baseline()
        self.run_reweighing()
        self.run_lfr()


class InprocessingModel:
    """
    A class that handles in-processing fairness mitigations:
    - Adversarial Debiasing
    - Exponentiated Gradient Reduction (EGR)
    - Metrics & plots
    """

    def __init__(self):
        self.input_file = "data/standard_df.csv"
        self.protected_attrs = PROTECTED_ATTRS
        self.target = "Target"

        self.df = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.baseline_df = None

    def load_and_split(self):
        self.df = read_csv(self.input_file)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            *_,
        ) = split_data(self.df, self.protected_attrs, self.target)

    def load_baseline(self):
        self.baseline_df = read_csv("output/csv/baseline_xgboost.csv")
        self.baseline_model = load_from_pickle("output/model/baseline_model.pkl")

    # Adversarial Debiasing
    def run_adversarial(self):
        disable_eager_execution()

        adv = AdversarialDebiasing(
            self.protected_attrs[:-1],
            scope_name="adversary",
            debias=True,
            random_state=0,
        )

        scorer = make_scorer(
            average_odds_scoring,
            scoring="average_odds",
            greater_is_better=False,
            protected_attrs=self.protected_attrs[0],
        )

        adv_grid = run_grid_search(
            adv, ADVERSARIAL_PARAMS, self.X_train, self.y_train, scorer, njobs=None
        )

        y_pred = adv_grid.predict(self.X_test)
        y_proba = adv_grid.predict_proba(self.X_test)[:, 1]

        adv_df = calculate_performance_and_fairness_metrics(
            self.y_test, y_pred, y_proba, self.protected_attrs
        )
        adv_df.to_csv("output/csv/adversarial_debiasing.csv", index=False)

        labels = ["Evaluation Metrics"] + [
            f"{prot_attr} Fairness Metrics" for prot_attr in self.protected_attrs
        ]
        metrics_list = build_metrics_list(
            adv_df,
            self.baseline_df,
            EVAL_METRICS,
            FAIRNESS_METRICS,
            "Adversarial Debiasing",
            self.protected_attrs,
        )
        plot_metrics_grid(
            metrics_list,
            labels,
            ["Adversarial Debiasing", "XGBoost"],
        )

    # Exponentiated Gradient Reduction
    def run_egr(self):
        np.random.seed(0)

        egr = ExponentiatedGradientReduction(
            self.protected_attrs,
            estimator=self.baseline_model,
            constraints="EqualizedOdds",
        )

        scorer = make_scorer(
            average_odds_scoring,
            scoring="EqualizedOdds",
            greater_is_better=False,
            protected_attrs=self.protected_attrs[0],
        )

        egr_grid = run_grid_search(egr, EGR_PARAMS, self.X_train, self.y_train, scorer)

        y_pred = egr_grid.predict(self.X_test)
        y_proba = egr_grid.predict_proba(self.X_test)[:, 1]

        egr_df = calculate_performance_and_fairness_metrics(
            self.y_test, y_pred, y_proba, self.protected_attrs
        )
        egr_df.to_csv("output/csv/egr.csv", index=False)

        labels = ["Evaluation Metrics"] + [
            f"{prot_attr} Fairness Metrics" for prot_attr in self.protected_attrs
        ]
        metrics_list = build_metrics_list(
            egr_df,
            self.baseline_df,
            EVAL_METRICS,
            FAIRNESS_METRICS,
            "EGR",
            self.protected_attrs,
        )
        plot_metrics_grid(metrics_list, labels, ["EGR", "XGBoost"])

    # Run all steps
    def run_inprocessing_models(self):
        self.load_and_split()
        self.load_baseline()
        self.run_adversarial()
        self.run_egr()


class PostProcessingModel:
    """
    A class that handles post-processing fairness mitigations:
    - Calibrated Equalized Odds
    - Reject Option Classification
    - Metrics & plots
    """

    def __init__(
        self,
    ):
        self.input_file = "data/standard_df.csv"
        self.protected_attrs = PROTECTED_ATTRS
        self.target = "Target"

        self.df = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.baseline_df = None

    def load_and_split(self):
        self.df = read_csv(self.input_file)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            *_,
        ) = split_data(self.df, self.protected_attrs, self.target)

    def load_baseline(self):
        self.baseline_df = read_csv("output/csv/baseline_xgboost.csv")
        self.baseline_model = load_from_pickle("output/model/baseline_model.pkl")

    # Calibrated Equalized Odds
    def run_calibrated_eq_odds(self):
        model = PostProcessingMeta(
            self.baseline_model,
            CalibratedEqualizedOdds(
                self.protected_attrs[0], cost_constraint="fnr", random_state=0
            ),
            random_state=0,
        )

        grid = model.fit(self.X_train, self.y_train)
        y_pred = grid.predict(self.X_test)
        y_proba = grid.predict_proba(self.X_test)[:, 1]

        cal_df = calculate_performance_and_fairness_metrics(
            self.y_test, y_pred, y_proba, self.protected_attrs
        )
        cal_df.to_csv("output/csv/calibrated_equalized_odds.csv", index=False)

        labels = ["Evaluation Metrics"] + [
            f"{prot_attr} Fairness Metrics" for prot_attr in self.protected_attrs
        ]
        metrics_list = build_metrics_list(
            cal_df,
            self.baseline_df,
            EVAL_METRICS,
            FAIRNESS_METRICS,
            "Calibrated Equalized Odds",
            self.protected_attrs,
        )
        plot_metrics_grid(
            metrics_list,
            labels,
            ["Calibrated Equalized Odds", "XGBoost"],
        )

    # Reject Option Classification
    def run_reject_option(self):
        model = PostProcessingMeta(
            self.baseline_model,
            RejectOptionClassifierCV(self.protected_attrs[0], scoring="average_odds"),
            random_state=0,
        )

        grid = model.fit(self.X_train, self.y_train)
        y_pred = grid.predict(self.X_test)
        y_proba = grid.predict_proba(self.X_test)[:, 1]

        rej_df = calculate_performance_and_fairness_metrics(
            self.y_test, y_pred, y_proba, self.protected_attrs
        )
        rej_df.to_csv("output/csv/reject_option_classification.csv", index=False)

        labels = ["Evaluation Metrics"] + [
            f"{prot_attr} Fairness Metrics" for prot_attr in self.protected_attrs
        ]
        metrics_list = build_metrics_list(
            rej_df,
            self.baseline_df,
            EVAL_METRICS,
            FAIRNESS_METRICS,
            "Reject Option Classification",
            self.protected_attrs,
        )
        plot_metrics_grid(
            metrics_list,
            labels,
            ["Reject Option Classification", "XGBoost"],
        )

    # Run all steps
    def run_postprocessing_models(self):
        self.load_and_split()
        self.load_baseline()
        self.run_calibrated_eq_odds()
        self.run_reject_option()
