# Vendored Tab-DDPM core

Source: https://github.com/yandex-research/tab-ddpm
Commit: `b476257dd460b778ba09eb97f7a51d6490fa17f8` (MIT; see LICENSE.md).

Only the diffusion, neural modules, and their tensor utilities are included.
Local changes to `gaussian_multinomial_diffsuion.py`:

- Allow `y_dist=None` to sample without target conditioning or dummy targets.
- Put absent numerical/categorical loss tensors on the input device.
- Raise on nonfinite samples before returning them; no filtering or retries.

The diffusion objective and denoiser are otherwise unchanged. Upstream data
loaders, evaluation scripts, and environment requirements are not used.
