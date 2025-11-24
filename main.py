from src.dataprocess import DataPreprocessorPipeline
import pandas as pd

import argparse
import subprocess
import sys


def main():

    data_processor = DataPreprocessorPipeline(df=df)
    data_processor.run_pipeline()


if __name__ == "__main__":
    main()
