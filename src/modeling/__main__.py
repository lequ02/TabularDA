"""Dispatch corrected modeling runs by dataset name."""

import argparse


CLASSIFICATION_DATASETS = {
    "adult", "census_kdd", "credit", "covertype",
    "mnist12", "mnist28",
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-name", required=True)
    args, _ = parser.parse_known_args()
    dataset = args.dataset_name.lower()

    if dataset in {"news", "california_housing"}:
        from . import regression_main

        entrypoint = regression_main
    elif dataset in CLASSIFICATION_DATASETS:
        from . import classification_main

        entrypoint = classification_main
    else:
        parser.error(
            "unsupported dataset; choose adult, census_kdd, credit, covertype, "
            "mnist12, mnist28, news, or california_housing"
        )

    entrypoint.main(entrypoint.parse_args())


if __name__ == "__main__":
    main()
