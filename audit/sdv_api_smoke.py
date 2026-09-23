"""Tiny SDV API check for the corrected environment; run on suitable compute."""

from importlib.metadata import version

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer


def main():
    real = pd.DataFrame({
        "feature": [0, 1] * 20,
        "Class": [0] * 38 + [1] * 2,
    })
    metadata = Metadata.detect_from_dataframe(real)
    metadata.update_column(table_name="table", column_name="Class", sdtype="categorical")
    metadata.validate()
    metadata.validate_data({"table": real})

    for synthesizer_type in (CTGANSynthesizer, TVAESynthesizer):
        synthesizer = synthesizer_type(
            metadata, epochs=1, batch_size=10, cuda=False, verbose=False
        )
        parameters = synthesizer.get_parameters()
        assert parameters["epochs"] == 1 and parameters["batch_size"] == 10
        synthesizer.fit(real)
        sample = synthesizer.sample(num_rows=5)
        assert list(sample.columns) == list(real.columns)
        assert len(sample) == 5
    print({name: version(name) for name in ("sdv", "ctgan", "rdt", "torch")})


if __name__ == "__main__":
    main()
