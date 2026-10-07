# Credit weighted macro-F1 pilot

Three seed-42 runs: real-only and synthetic-only CTGAN full-table+DNN / features-only+DNN. Existing generator, labeler, and input artifacts are reused.

Class-weighted BCEWithLogitsLoss (negative/positive training counts), real-development macro-F1 checkpoint selection, strict probability > 0.5. Existing Credit architecture, split, scaling, batch order, batch 128, learning rate 0.001, maximum 100 epochs, patience 30.

Original results remain in corrected_v2; these outputs are separate. manifest.json records protocol and input hashes; run records contain per-arm weights and selected development diagnostics. completed.json requires all three runs. Failures surface in queue/arm logs; no retries or model substitutions.

Both objective and checkpoint selection changed. Evaluate minority precision/recall and F1, not merely positive predictions. The real test set has only 10 fraud cases, so also inspect confusion counts.
