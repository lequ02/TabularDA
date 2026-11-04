"""Unconditional Tab-DDPM on exactly the supplied columns; reuse our labelers."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from tabddpm_core.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from tabddpm_core.modules import MLPDiffusion


UPSTREAM_COMMIT = 'b476257dd460b778ba09eb97f7a51d6490fa17f8'


class TabDDPM:
    def __init__(self, *, seed=42, steps=20_000, batch_size=500,
                 num_timesteps=1000, sample_batch_size=500, device='cuda'):
        if min(steps, batch_size, num_timesteps, sample_batch_size) <= 0:
            raise ValueError('Tab-DDPM budgets and batch sizes must be positive')
        self.seed = seed
        self.steps = steps
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.sample_batch_size = sample_batch_size
        self.device = device

    def _diffusion(self):
        model = MLPDiffusion(
            d_in=len(self.numerical) + sum(self.cardinalities),
            num_classes=0, is_y_cond=False,
            rtdl_params={'d_layers': [256, 256], 'dropout': 0.0},
        )
        return GaussianMultinomialDiffusion(
            num_classes=np.array(self.cardinalities or [0]),
            num_numerical_features=len(self.numerical), denoise_fn=model,
            num_timesteps=self.num_timesteps, gaussian_loss_type='mse',
            scheduler='cosine', device=torch.device(self.device),
        ).to(self.device)

    def fit(self, data, categorical_columns):
        if data.empty or not data.columns.is_unique or data.isna().any().any():
            raise ValueError('Tab-DDPM requires a nonempty prepared table without missing values')
        if not set(categorical_columns).issubset(data.columns):
            raise ValueError('Categorical columns must belong to the supplied fit table')
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.columns = list(data.columns)
        self.categorical = [c for c in self.columns if c in categorical_columns]
        self.numerical = [c for c in self.columns if c not in self.categorical]
        parts = []
        self.discrete_values = {}
        if self.numerical:
            values = data[self.numerical].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite numerical training values')
            self.normalizer = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(len(data) // 30, 1000), 10),
                subsample=1_000_000_000, random_state=self.seed,
            )
            parts.append(self.normalizer.fit_transform(values).astype(np.float32))
            # Preserve upstream rounding for small integer-valued numerical domains.
            for i, column in enumerate(self.numerical):
                unique = np.unique(values[:, i])
                if len(unique) <= 32 and np.equal(unique, np.round(unique)).all():
                    self.discrete_values[column] = unique
        self.cardinalities = []
        if self.categorical:
            self.encoder = OrdinalEncoder(dtype=np.int64)
            parts.append(self.encoder.fit_transform(data[self.categorical]).astype(np.float32))
            self.cardinalities = [len(c) for c in self.encoder.categories_]
        features = torch.from_numpy(np.concatenate(parts, axis=1))
        loader = DataLoader(TensorDataset(features), batch_size=self.batch_size, shuffle=True)
        diffusion = self._diffusion()
        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=0.001, weight_decay=0.00001)
        history = []
        step = 0
        diffusion.train()
        while step < self.steps:
            for (batch,) in loader:
                optimizer.zero_grad()
                multi, gauss = diffusion.mixed_loss(batch.to(self.device), {})
                loss = multi + gauss
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite Tab-DDPM loss at step {step}')
                loss.backward()
                optimizer.step()
                step += 1
                for group in optimizer.param_groups:
                    group['lr'] = 0.001 * (1 - step / self.steps)
                if step == 1 or step % 100 == 0 or step == self.steps:
                    history.append((step, float(multi.detach()), float(gauss.detach())))
                    print(f'Tab-DDPM step {step}/{self.steps}, loss={float(loss.detach()):.6f}', flush=True)
                if step == self.steps:
                    break
        self.weights = {k: v.detach().cpu() for k, v in diffusion._denoise_fn.state_dict().items()}
        self.loss_history = pd.DataFrame(history, columns=['step', 'multinomial_loss', 'gaussian_loss'])

    def sample(self, num_rows):
        if num_rows <= 0:
            raise ValueError('Sample row count must be positive')
        # All labelers receive the same X-only sample, including after save/load.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        diffusion = self._diffusion()
        diffusion._denoise_fn.load_state_dict(self.weights)
        diffusion.eval()
        values, _ = diffusion.sample_all(num_rows, self.sample_batch_size, y_dist=None)
        values = values.numpy()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite Tab-DDPM sample')
        result = pd.DataFrame(index=range(num_rows))
        n_num = len(self.numerical)
        if n_num:
            decoded = self.normalizer.inverse_transform(values[:, :n_num])
            for i, column in enumerate(self.numerical):
                result[column] = decoded[:, i]
                if column in self.discrete_values:
                    unique = self.discrete_values[column]
                    result[column] = unique[np.abs(decoded[:, i, None] - unique).argmin(axis=1)]
        if self.categorical:
            decoded = self.encoder.inverse_transform(values[:, n_num:].astype(np.int64))
            for i, column in enumerate(self.categorical):
                result[column] = decoded[:, i]
        return result.loc[:, self.columns]

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        self.loss_history.to_csv(Path(path).with_suffix('.loss.csv'), index=False)

    def get_parameters(self):
        return dict(upstream_commit=UPSTREAM_COMMIT, seed=self.seed, steps=self.steps,
                    batch_size=self.batch_size, num_timesteps=self.num_timesteps,
                    sample_batch_size=self.sample_batch_size, device=self.device,
                    is_y_cond=False, d_layers=[256, 256], dim_t=128, dropout=0.0,
                    lr=0.001, weight_decay=0.00001, scheduler='cosine',
                    gaussian_loss_type='mse', checkpoint='final', sampler='ancestral')


def create_tabddpm_data(job, steps):
    """Fit full/X-only tables, then invoke the existing real-data labeling path."""
    from commons.onehot import onehot
    from modeling import constants
    from synthesizer import _save_synthesis_provenance, _save_synthetic_quality
    from synthesizer import synthesize_comparison_from_trained_model

    manifest = Path(job.paths['data_dir']) / 'split_manifest.json'
    if not manifest.is_file():
        job.prepare_train_test()
    split = json.loads(manifest.read_text())
    if (split['dataset'], split['seed']) != (job.ds_name, job.seed):
        raise ValueError('Prepared split manifest does not match the dataset/seed')
    for partition in ('train', 'dev', 'test'):
        for view in ('raw', 'onehot'):
            path = Path(job.paths['data_dir']) / constants.split_name(job.ds_name, job.seed, partition, view)
            digest = hashlib.sha256()
            with path.open('rb') as file:
                for block in iter(lambda: file.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != split['files'][partition][view]['sha256']:
                raise ValueError(f'Prepared split hash mismatch: {path}')
    xtrain, ytrain, target, categorical = job.read_train_data()
    x_encoded, _ = onehot(xtrain, xtrain, categorical)
    labelers = ['pca_gmm', 'rf', 'xgb', 'dnn']
    if job.is_classification:
        labelers = ['gaussianNB', 'categoricalNB'] + labelers
    dev = job.read_dev_data()
    for source in ('full', 'xonly'):
        fit_table = pd.concat([xtrain, ytrain], axis=1) if source == 'full' else xtrain
        fit_categories = list(categorical)
        if source == 'full' and job.is_classification:
            fit_categories.append(target)
        model_path = Path(job.paths['synthesizer_dir']) / constants.generator_name(
            job.ds_name, job.seed, 'tabddpm', source)
        if model_path.exists():
            raise FileExistsError(f'Tab-DDPM generator already exists; preserve it: {model_path}')
        generator = TabDDPM(seed=job.seed, steps=steps)
        generator.fit(fit_table, fit_categories)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        generator.save(model_path)
        _save_synthesis_provenance(generator, fit_table, job.sample_size_to_synthesize,
                                  str(model_path), job.seed,
                                  metadata={'columns': generator.columns,
                                            'categorical_columns': generator.categorical})
        sample = generator.sample(job.sample_size_to_synthesize)
        _, features = onehot(xtrain, sample.drop(columns=[target]) if source == 'full' else sample,
                             categorical)
        features = features.reindex(sorted(features.columns), axis=1)
        if source == 'full':
            baseline = pd.concat([features, sample[target]], axis=1)
            baseline_path = Path(job.paths['data_dir']) / constants.synthetic_name(
                job.ds_name, job.seed, 'tabddpm')
            baseline.to_csv(baseline_path, index=False)
            _save_synthetic_quality(baseline, x_encoded, ytrain, target,
                                   job.sample_size_to_synthesize, str(baseline_path),
                                   job.is_classification)
        # Persist X once; no resampling between supervised labelers.
        features.to_csv(model_path.with_suffix('.sample.csv'), index=False)
        for labeler in labelers:
            label = {'gaussianNB': 'gaussian', 'categoricalNB': 'categorical'}.get(labeler, labeler)
            method = 'tabddpm_' + ('compare_' if source == 'full' else '') + label
            csv_path = Path(job.paths['data_dir']) / constants.synthetic_name(job.ds_name, job.seed, method)
            synthesize_comparison_from_trained_model(
                xtrain, ytrain, categorical, target_name=target,
                sample_size=job.sample_size_to_synthesize, target_synthesizer=labeler,
                numerical_columns_pca_gmm=job.numerical_cols_pca_gmm,
                csv_file_name=str(csv_path), is_classification=job.is_classification,
                seed=job.seed, dnn_dev_data=dev, dataset_name=job.ds_name,
                synthetic_features=features,
            )
