# Census KDD weighted macro-F1 evaluation

106 planned configurations: seeds 42/43, real-only plus CTGAN/TVAE full-generated, full-relabelled, and features-only-relabelled arms under synthetic and mix training.

Three seed-42 pilot results were copied with verified artifact hashes and explicit source-record provenance; 103 runs are new. Original corrected_v2 and pilot results remain preserved.

BCEWithLogitsLoss, positive weight = actual training negatives / positives, real-development macro-F1 checkpoint selection, strict probability > 0.5, unshuffled batches. Existing architecture, splits, scaling, learning rate 0.001, batch 128, 100 epochs, patience 30. Mix uses every real training row plus 100,000 synthetic rows.

manifest.json records the exact protocol, inputs, configurations, and pilot imports. status.json reports current completed/failed/running counts. completed.json exists only after all 106 records pass verification. Logs retain actual failures; there are no automatic retries.

Binary F1 and macro F1 remain distinct. Weighted losses differ across arms and are not common unweighted BCE. Both weighting and checkpoint selection changed; do not attribute improvements to weighting alone.
