import sys
import os
import pandas as pd
from onehot import *
from synthesizer import get_metadata

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import *
from sdmetrics.reports.single_table import QualityReport



def create_report(real_data, synthetic_data, metadata, filepath='../data/report.pkl', verbose=False):
    report = QualityReport()
    report.generate(real_data, synthetic_data, metadata)
    report.save(filepath=filepath)

    if verbose:
        print("Score: ")
        print(report.get_score())
        print("\nProperties: ")
        print(report.get_properties())
        print("\nColumn Shapes: ")
        print(report.get_details(property_name='Column Shapes'))
        print("\nColumn Pair Trends: ")
        print(report.get_details(property_name='Column Pair Trends'))
    print(f"\nReport saved at: {filepath}")

    return report


def report_census():
    x, y = load_census()
    y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})
    real_census = pd.concat([x, y], axis=1)

    metadata = get_metadata(real_census).to_dict()
    print("Census metadata")
    print(metadata)

    synthetic_sdv_categorical_1mil = pd.read_csv('../data/census/census_sdv_categorical_1mil.csv', index_col=0) 
    # synthetic_data.drop(columns=['income.1'], inplace=True)
    report1 = create_report(real_census, synthetic_sdv_categorical_1mil, metadata, filepath='../data/census/census_sdv_categorical_1mil_quality_report.pkl', verbose=True)

    synthetic_sdv_gaussian_1mil = pd.read_csv('../data/census/census_sdv_gaussian_1mil.csv', index_col=0) 
    report2 = create_report(real_census, synthetic_sdv_gaussian_1mil, metadata, filepath='../data/census/census_sdv_gaussian_1mil_quality_report.pkl', verbose=True)

    synthetic_sdv_1mil = pd.read_csv('../data/census/census_sdv_1mil.csv', index_col=0) 
    report3 = create_report(real_census, synthetic_sdv_1mil, metadata, filepath='../data/census/census_sdv_1mil_quality_report.pkl', verbose=True)


def main():
    report_census()

if __name__ == '__main__':
    main()