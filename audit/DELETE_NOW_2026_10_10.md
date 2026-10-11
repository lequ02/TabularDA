# Verified local deletion list

Checked 2026-10-10T23:43:12.764698-05:00 (Chicago). No files were deleted, moved, or modified by this verification, apart from writing this review and its exact-path CSV. No experiments or model loading were performed.

This is a conservative list of files that can be removed without losing unique research content, provided their listed retained counterparts and preservation manifests remain. It is not a blanket deletion classification for everything outside the selected final results.

| Category | Files | Bytes | GiB | Why deletion is safe |
| --- | ---: | ---: | ---: | --- |
| BYTE_IDENTICAL_CHECKPOINT_COPY | 4 | 471,244,705 | 0.439 | Fresh SHA256 checks match the retained adjacent checkpoint exactly; source/notebook searches found no executable reference to the old model paths. |
| RESTORABLE_NEWLINE_BACKUP | 216 | 714,359,545 | 0.665 | All 216 files contain the same run records as their retained canonical counterparts. Converting retained LF text to CRLF freshly reproduced each original backup byte-for-byte. |
| REDUNDANT_REPORT_ZIP | 11 | 1,330,885,722 | 1.239 | All 2,751 member mappings resolve to 749 retained files with freshly verified exact hashes; historical snapshots and member provenance are preserved. |

Total: 231 files, 2,516,489,972 bytes (2.344 GiB).

The complete filenames, hashes, keeper paths, and restoration rules are in [delete_now_verified_2026_10_10.csv](delete_now_verified_2026_10_10.csv).

## Four duplicate checkpoint files

- Remove `sdv trained model/mnist12/old_split/mnist12_synthesizer_onlyX.pkl`; retain `sdv trained model/mnist12/mnist12_synthesizer_onlyX.pkl`.
- Remove `sdv trained model/mnist12/old_split/mnist12_synthesizer.pkl`; retain `sdv trained model/mnist12/mnist12_synthesizer.pkl`.
- Remove `sdv trained model/covertype/old/covertype_synthesizer_onlyX.pkl`; retain `sdv trained model/covertype/covertype_synthesizer_onlyX.pkl`.
- Remove `sdv trained model/covertype/old/covertype_synthesizer.pkl`; retain `sdv trained model/covertype/covertype_synthesizer.pkl`.

## 216 redundant loose backup records

Only the 216 explicitly listed `*.run.json` files under `.cache/remote_intrusion/mix_refresh_backup_20261005/output/`. Do not remove the entire backup tree. The canonical records and the original/keeper hash mapping remain required.

## Eleven redundant historical report ZIPs

- `.cache/remote_intrusion/comparison_before_refresh_20261008_170614/.cache/remote_intrusion/latest_comparison.zip`
- `.cache/remote_intrusion/comparison_before_refresh_20261008_170614/.cache/remote_intrusion/latest_comparison_mix.zip`
- `.cache/remote_intrusion/comparison_before_refresh_20261009_100501/.cache/remote_intrusion/latest_comparison.zip`
- `.cache/remote_intrusion/comparison_before_refresh_20261009_100501/.cache/remote_intrusion/latest_comparison_mix.zip`
- `.cache/remote_intrusion/comparison_before_refresh_20261009_214058/.cache/remote_intrusion/latest_comparison.zip`
- `.cache/remote_intrusion/comparison_before_refresh_20261009_214058/.cache/remote_intrusion/latest_comparison_mix.zip`
- `.cache/remote_intrusion/comparison_before_refresh_20261010_115649/.cache/remote_intrusion/latest_comparison.zip`
- `.cache/remote_intrusion/comparison_before_refresh_20261010_115649/.cache/remote_intrusion/latest_comparison_mix.zip`
- `.cache/remote_intrusion/comparison_before_removal_20261008_172619/latest_comparison.zip`
- `.cache/remote_intrusion/comparison_before_removal_20261008_172619/latest_comparison_mix.zip`
- `.cache/remote_intrusion/mix_report_before_refresh_20261005_193014/latest_comparison_mix.zip`

Preserve `audit/experiment_cleanup_archive_member_manifest_2026_10_10.json`, `audit/experiment_cleanup_archive_snapshots_20261010/`, every member manifest `keep_file`, and this deletion list. Every original member and snapshot is retained byte-for-byte. Repacking may change ZIP ordering, compression, timestamps, and the whole-ZIP hash; exact original ZIP container bytes are not promised. If retaining the historical container itself is a requirement, exclude these eleven ZIPs and the remaining 220-file list frees 1.104 GiB.

## Dependency review and exclusions

Targeted fresh searches of local Python sources, notebooks, R Markdown, and JSON found two refresh-history JSON references to dated backup directories. These describe backup locations, rather than consume the eleven ZIPs. Keep those directories and their other contents. The executable comparison verifier reads the October 7 ZIPs and current ZIPs, which are excluded.

Do not delete either October 7 comparison ZIP; the two current `.cache/remote_intrusion/latest_comparison*.zip` files; any other file in the dated backup directories; pilot results or decision studies; unique failed-run evidence; original source data or ambiguous legacy CSVs; selected checkpoints; the 406 copied frozen report records until their readers are changed; notebook checkpoints with unreviewed edits; or whole `old`, `old_split`, `.cache`, or `audit` directories.

This review does not approve deletion of the 640 CSV duplicate candidates. Those require a separate check of final deliverables, standalone packages, historical paths, and rename mappings. Audit generators and intermediate inventories may eventually be removed after the retained conclusions/provenance are consolidated; they are not included here.

This list represents the verified filesystem state at the recorded time. If deletion occurs later, recheck the listed hashes and keeper existence first because other chats are working in this repository.

## Executed deletion

Completed 2026-10-10T23:52:25.7600452-05:00 (Chicago), following the user's explicit instruction to delete all 231 listed files. Removed 231 files totaling 2516489972 bytes (2.344 GiB). All authorized paths were confirmed absent, all verified retained copies remained present and unchanged, and the current/October 7 protected comparison ZIPs remained unchanged. No directories were removed. The per-file deletion receipt is [deletion_receipt_2026_10_10.jsonl](deletion_receipt_2026_10_10.jsonl).
