"""Unconditional Tab-DDPM on exactly the supplied columns; reuse our labelers."""

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
    def __init__(self, *, seed=42, steps=None, epochs=None, batch_size=500,
                 num_timesteps=1000, sample_batch_size=500, device='cuda'):
        if steps is not None and epochs is not None:
            raise ValueError('Choose either Tab-DDPM steps or epochs, not both')
        if steps is None and epochs is None:
            steps = 20_000
        budget = epochs if epochs is not None else steps
        if not isinstance(budget, int) or min(budget, batch_size, num_timesteps, sample_batch_size) <= 0:
            raise ValueError('Tab-DDPM budgets and batch sizes must be positive')
        self.seed = seed
        self.steps = steps
        self.epochs = epochs
        self.training_rows = None
        self.updates_per_epoch = None
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
        self.training_rows = len(data)
        self.updates_per_epoch = len(loader)
        if self.epochs is not None:
            self.steps = self.epochs * self.updates_per_epoch
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
                    epochs=self.epochs, training_rows=self.training_rows,
                    updates_per_epoch=self.updates_per_epoch,
                    batch_size=self.batch_size, num_timesteps=self.num_timesteps,
                    sample_batch_size=self.sample_batch_size, device=self.device,
                    is_y_cond=False, d_layers=[256, 256], dim_t=128, dropout=0.0,
                    lr=0.001, weight_decay=0.00001, scheduler='cosine',
                    gaussian_loss_type='mse', checkpoint='final', sampler='ancestral')


def create_tabddpm_data(job, steps=None, *, epochs=None):
    """Fit full/X-only tables, then invoke the existing real-data labeling path."""
    from commons.onehot import onehot
    from commons.log_target import log_target, inverse_log_target
    from modeling import constants
    from synthesizer import _save_synthesis_provenance, _save_synthetic_quality, file_sha256
    from synthesizer import synthesize_comparison_from_trained_model

    job.ensure_prepared()
    for folder in ('data_dir', 'synthesizer_dir'):
        existing = list(Path(job.paths[folder]).glob(f'{job.ds_name}_seed{job.seed}_tabddpm_*'))
        if existing:
            raise FileExistsError(f'Tab-DDPM artifacts already exist; preserve them: {existing[0]}')
    xtrain, ytrain, target, categorical = job.read_train_data()
    x_encoded, _ = onehot(xtrain, xtrain, categorical)
    labelers = ['pca_gmm', 'rf', 'xgb', 'dnn']
    if job.is_classification:
        labelers = ['gaussianNB', 'categoricalNB'] + labelers
    dev = job.read_dev_data()
    for source in ('full', 'xonly'):
        fit_table = pd.concat([xtrain, ytrain], axis=1) if source == 'full' else xtrain
        fit_transform = job.target_transform if source == 'full' else 'raw'
        if fit_transform == 'log':
            fit_table[target] = log_target(ytrain)
        fit_categories = list(categorical)
        if source == 'full' and job.is_classification:
            fit_categories.append(target)
        model_path = Path(job.paths['synthesizer_dir']) / constants.generator_name(
            job.ds_name, job.seed, 'tabddpm', source)
        generator = TabDDPM(seed=job.seed, steps=steps, epochs=epochs)
        generator.fit(fit_table, fit_categories)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        generator.save(model_path)
        _save_synthesis_provenance(generator, fit_table, job.sample_size_to_synthesize,
                                  str(model_path), job.seed,
                                  metadata={'columns': generator.columns,
                                            'categorical_columns': generator.categorical},
                                  target_transform=fit_transform)
        sample = generator.sample(job.sample_size_to_synthesize)
        if fit_transform == 'log':
            sample[target] = inverse_log_target(sample[target])
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
                                   job.is_classification, target_transform=job.target_transform)
        # Persist X once; no resampling between supervised labelers.
        feature_path = model_path.with_suffix('.sample.csv')
        features.to_csv(feature_path, index=False)
        feature_source = dict(path=str(feature_path), sha256=file_sha256(feature_path),
                              generator_path=str(model_path), generator_sha256=file_sha256(model_path),
                              seed=job.seed, rows=len(features))
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
                synthetic_features=features, target_transform=job.target_transform,
                feature_source=feature_source,
            )
