from src.dataprocess import DataPreprocessorPipeline
from src.bias_detection import PreModelingBias, PostModelingBias
from src.constants import PROTECTED_ATTRS
from src.explainability import Explainability
from src.bias_mitigation import (
    PreprocessingModel,
    InprocessingModel,
    PostProcessingModel,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Fairness ML and Bias Mitigation Pipelines"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=[
            "preprocess",
            "pre-modeling-bias",
            "post-modeling-bias",
            "preprocessing-models",
            "inprocessing-models",
            "postprocessing-models",
            "explainability",
        ],
        required=True,
        help="Choose which pipeline to run: fairness (preprocess, pre-modeling-bias, post-modeling-bias), bias mitigation (preprocessing-models, inprocessing-models, postprocessing-models), or explainability",
    )
    args = parser.parse_args()

    if args.pipeline in ["preprocess", "pre-modeling-bias", "post-modeling-bias"]:
        if args.pipeline == "preprocess":
            data_processor = DataPreprocessorPipeline()
            data_processor.run_pipeline()
        elif args.pipeline == "pre-modeling-bias":
            pre_model_bias = PreModelingBias()
            pre_model_bias.correlation_analysis()
            pre_model_bias.plot_kde_distributions()
        elif args.pipeline == "post-modeling-bias":
            post_model_bias = PostModelingBias()
            fairness_df, metrics = post_model_bias.train_and_evaluate()
    elif args.pipeline in [
        "preprocessing-models",
        "inprocessing-models",
        "postprocessing-models",
        "explainability",
    ]:
        if args.pipeline == "preprocessing-models":
            model = PreprocessingModel()
            model.run_preprocessing_models()
        elif args.pipeline == "inprocessing-models":
            model = InprocessingModel()
            model.run_inprocessing_models()
        elif args.pipeline == "postprocessing-models":
            model = PostProcessingModel()
            model.run_postprocessing_models()
        elif args.pipeline == "explainability":
            explainer = Explainability()
            explainer.run_pipeline()


if __name__ == "__main__":
    main()
