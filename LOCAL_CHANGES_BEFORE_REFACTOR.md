# Local changes and checkpoint before major refactor

Checkpoint tag: `checkpoint/pre-major-refactor-2026-09-18` in the main repository, GLRM and pyglrm.
Original parent commit: `248cb9b3cdb3ae5ef6783b20c0edbffa8119f847`.

| Group | Preserved local changes |
|---|---|
| Synthesis routing | Normalize CTGAN/TVAE names, update artifact paths and trained-model dispatch in CreateSyntheticData.py |
| Synthesis validation | Check encoded feature column names and order before label synthesis; remove duplicated comments in synthesizer.py |
| Credit exploration | 19 added EDA_Credit.ipynb cells and saved outputs |
| Results processing | New post_processing.ipynb notebook |
| GLRM | Compatibility updates in five source files; reconstruction metrics; new glrm_df.py dataframe wrapper |
| SDGym | 19 dataset metadata and README files |
| Audit | Code/data audit, historical reconstruction, research result review, downloaded evidence, CSV tables and experiment logs |
| Presentations | BOLDED.png, Poster.pdf, Poster.ppt, Poster_final.pdf, Untitled PNG, Xsyn.png, Xsyn2.png, three flowchart PNGs, five preview PNGs in tmp/ |
| Archived MNIST | Two ensemble_results.txt files and the 164.93 MiB model ZIP stored through Git LFS |
| Tracked caches | 11 modified Python bytecode files in a separate snapshot commit |
| Checkpoint housekeeping | Exclude installed node/egg dependencies; explicitly include audit CSV evidence; add archive LFS rule and this inventory |

All existing research source changes were preserved as found. Syntax validation passed for the two changed synthesis modules and seven GLRM modules; both notebooks contain valid JSON. Training and experiment reruns were not performed. Known correctness issues remain documented in audit/AUDIT.md.

Installed dependencies in audit/results_review/node_modules (a directory link) and pyglrm/.eggs are excluded from commits. Raw Drive HTML snapshots were subsequently omitted to remove embedded key strings; their parsed listings and research files are retained. Failed Git packing and LFS staging encountered memory/disk limits; stale temporary Git files were removed and the checkpoint commits were completed with automatic packing disabled for the commit commands.

Existing ignored raw CSV datasets and pickle models remain on disk, outside this Git checkpoint. The old MNIST split/model folders total about 3.8 GiB, including the ZIP now captured through Git LFS. Audit CSV evidence is explicitly included. Other pre-existing ignored files elsewhere in the repository are also outside the checkpoint.

The original checkpoint and nested backups were pushed to lequ02/TabularDA. A sanitized replacement is prepared locally; see audit/SECRET_SCAN.md for its scope and publication status. The repository already used GLRM/pyglrm Git links without a .gitmodules file; preserve those local repositories with the parent repository. The tag records the parent commit, both nested repository states and the archive LFS pointer.

## Main repository commits

| Commit | Change |
|---|---|
| `bff2366` | Normalize feature synthesizer names and artifact paths |
| `497c0c3` | Validate encoded feature columns before label synthesis |
| `bd48209` | Preserve credit synthesis exploratory notebook |
| `72e9c10` | Preserve result post-processing notebook |
| `65a9e7e` | Exclude installed research dependencies |
| `db1f5ff` | Preserve SDGym dataset metadata |
| `555ee10` | Preserve code and data audit evidence |
| `edf5728` | Preserve historical research result reconstruction |
| `c92edf0` | Preserve research results review and supporting evidence |
| `8f49fdb` | Preserve research posters and supporting figures |
| `c9e105b` | Checkpoint archived MNIST ensemble logs and model bundle |
| `3fe2ea7` | Checkpoint existing tracked Python bytecode |
| `da84027` | Preserve nested GLRM repository checkpoints |
| `0e9a1b8` | Preserve audit CSV tables and experiment logs |

A final inventory commit adds this document; the named checkpoint tag includes that commit.

## Complete main repository file inventory

Status is relative to the original parent commit: M = modified; A = added. Checkpoint housekeeping changes are included.

```text
M	.gitattributes
M	.gitignore
A	BOLDED.png
M	GLRM
A	Poster.pdf
A	Poster.ppt
A	Poster_final.pdf
A	SDGym/datasets/adult/metadata_v0.json
A	SDGym/datasets/adult/metadata_v1.json
A	SDGym/datasets/alarm/metadata_v0.json
A	SDGym/datasets/alarm/metadata_v1.json
A	SDGym/datasets/census/metadata_v0.json
A	SDGym/datasets/census/metadata_v1.json
A	SDGym/datasets/child/metadata_v0.json
A	SDGym/datasets/child/metadata_v1.json
A	SDGym/datasets/covtype/metadata_v0.json
A	SDGym/datasets/covtype/metadata_v1.json
A	SDGym/datasets/expedia_hotel_logs/README.txt
A	SDGym/datasets/expedia_hotel_logs/metadata_v0.json
A	SDGym/datasets/expedia_hotel_logs/metadata_v1.json
A	SDGym/datasets/insurance/metadata_v0.json
A	SDGym/datasets/insurance/metadata_v1.json
A	SDGym/datasets/intrusion/metadata_v0.json
A	SDGym/datasets/intrusion/metadata_v1.json
A	SDGym/datasets/news/metadata_v0.json
A	SDGym/datasets/news/metadata_v1.json
A	Untitled (1920 x 678 px).png
A	Xsyn.png
A	Xsyn2.png
A	audit/AUDIT.md
A	audit/evidence.jsonl
A	audit/probes.jsonl
A	audit/reconstruction/328_project_pdf.txt
A	audit/reconstruction/April02_tidy.csv
A	audit/reconstruction/April29_macro_max_table.csv
A	audit/reconstruction/April29_tidy.csv
A	audit/reconstruction/DEDUPLICATED_RESULTS.md
A	audit/reconstruction/Mar23_tidy.csv
A	audit/reconstruction/Math328 Presentation_pdf.txt
A	audit/reconstruction/Poster_final_text.txt
A	audit/reconstruction/Poster_text.txt
A	audit/reconstruction/RECONSTRUCTION.md
A	audit/reconstruction/aggregation_blob.json
A	audit/reconstruction/aggregation_run_trace.txt
A	audit/reconstruction/analysisApril02_pdf.txt
A	audit/reconstruction/analysis_inventory.json
A	audit/reconstruction/analysis_numbers.json
A	audit/reconstruction/analysis_numbers.txt
A	audit/reconstruction/analysis_pdf.txt
A	audit/reconstruction/april29_commit.json
A	audit/reconstruction/check_statistics.R
A	audit/reconstruction/deduplicate_statistics.R
A	audit/reconstruction/deduplicated_dataset_differences.csv
A	audit/reconstruction/deduplicated_method_scores.csv
A	audit/reconstruction/deduplicated_statistics.txt
A	audit/reconstruction/fetch_evidence.py
A	audit/reconstruction/final_328_project_pdf.txt
A	audit/reconstruction/github_blobs/00556d5a7b6187948043aa6e738508d0a7cc93e0
A	audit/reconstruction/github_blobs/0292b89de35c959e92b2e2803ae9ca81727265c5
A	audit/reconstruction/github_blobs/06ce963b4fde760df8aa22bce84c52cfd9d4be2b
A	audit/reconstruction/github_blobs/08abda1eb9e704e263420910b1f58c1e1aee990a
A	audit/reconstruction/github_blobs/09625b20eea6ae82c0cb877d6901431f23ad5fe6
A	audit/reconstruction/github_blobs/1155667e306cc38c201301cacb2a6eef898b377b
A	audit/reconstruction/github_blobs/11c7c91a7f021f9d1ed9dfd31f75240d2ccc91e9
A	audit/reconstruction/github_blobs/127f353175227d86a908aded8eaad5bb598ab907
A	audit/reconstruction/github_blobs/129cffb3c7eba4c5fc9bcbe07ffbb729ca2df7ea
A	audit/reconstruction/github_blobs/12bd4457b373e335a07b8c503b9f187cd661e207
A	audit/reconstruction/github_blobs/13edc4512525505b6ebb506213f410dadb8b6421
A	audit/reconstruction/github_blobs/17a4555c8b445f1c283ae90114f50337163a2bfa
A	audit/reconstruction/github_blobs/1816c56e4e6a9b0403e1300f0e9d160d1264dd0c
A	audit/reconstruction/github_blobs/1a5777a8d3ad268c45fbf499d6e587bcd2a51a79
A	audit/reconstruction/github_blobs/1a9608e67e0990cb233a7665fbc423473d51378a
A	audit/reconstruction/github_blobs/1ca6c6f78352cea17e69369d5a957a4ae28423eb
A	audit/reconstruction/github_blobs/1e4d13b3e397884e411c09ed8be7dab79d5c7e01
A	audit/reconstruction/github_blobs/1eff97164b23784e77c9731fa787d2f33d33ba31
A	audit/reconstruction/github_blobs/1f1c0591be11739141e798735af9456e20688c2a
A	audit/reconstruction/github_blobs/2188cc6f2267f98019fe37cd8d2fcf6315a877de
A	audit/reconstruction/github_blobs/21effc48bb8e7b3bb6f91293d826b5aa7eba778e
A	audit/reconstruction/github_blobs/22922e537c20e5e89bf4fd45a1ceaa04920cb461
A	audit/reconstruction/github_blobs/240d916c2a3f171296419baa0893f583aaf53538
A	audit/reconstruction/github_blobs/27c596ecdd02ffa7287391c9043f817c346d13cd
A	audit/reconstruction/github_blobs/2a87a4061df36656142e84edc52815d5ffb07515
A	audit/reconstruction/github_blobs/2cf454fabd4d771c97019b9043bfaf59ee41672b
A	audit/reconstruction/github_blobs/2d2199a7ef429b983a830fe7d24f296fe5e82421
A	audit/reconstruction/github_blobs/2dc27476b60e4a9fe54ec221f4414548fd178bfd
A	audit/reconstruction/github_blobs/2e68879d6bad85db3421a275fc2a215687fe1e15
A	audit/reconstruction/github_blobs/3018d1bf1375bd2b9a591449a837f886bcaa4f49
A	audit/reconstruction/github_blobs/301fd9ee15bd444d92ea6ddcc1272f40f49ce12a
A	audit/reconstruction/github_blobs/30e7a7280e4b1433325fb4824905b647560578e3
A	audit/reconstruction/github_blobs/3234c4e5158c1a2515329cefe54c6de3fc9bbeb9
A	audit/reconstruction/github_blobs/32bef3b9c59b0c3a08be13ff1291e7e459bf2c76
A	audit/reconstruction/github_blobs/32e9e58467967ef0b10a22c6879843e122437408
A	audit/reconstruction/github_blobs/369637a7eb06beabf22ba7eef3896c2fed6ad069
A	audit/reconstruction/github_blobs/372fda3d03bf7d91f1037345ae127fd434c5bced
A	audit/reconstruction/github_blobs/37671c1bd41140527a08f1026e5454e19a6c5f3d
A	audit/reconstruction/github_blobs/37fec4184dc0005fc1a9b4ba6ecfae3d5fb0a35a
A	audit/reconstruction/github_blobs/38da86ddbed455738b12788a8e2d81388ca0d16e
A	audit/reconstruction/github_blobs/39e9646e1cebc2e3fec921c62c7acbd84ec1c999
A	audit/reconstruction/github_blobs/3b20bf9aeccd99e597cf6d02ee047610a92420f1
A	audit/reconstruction/github_blobs/3c45b9c4b8d5104059c1d476ced634c31a233eb8
A	audit/reconstruction/github_blobs/3d2f586fd066fb3a4a840b362666d4738d117f32
A	audit/reconstruction/github_blobs/41904b0533fba0c83d40dae5d7b18bac81cbb770
A	audit/reconstruction/github_blobs/436d4756138214c736cac17010822d8f1c4d945e
A	audit/reconstruction/github_blobs/437d9b0be36729c9f3cdce9d081f15a18846afb6
A	audit/reconstruction/github_blobs/45164c2f01fa6a222ae12209d00efed4d377bd06
A	audit/reconstruction/github_blobs/4600d7e766ae8b05fe554a2a732c3d3c9e22b140
A	audit/reconstruction/github_blobs/474882246d43339d0d772c11fafcd8c3170a395b
A	audit/reconstruction/github_blobs/498ec37598bd8fef388d7658708dc6fe7c9ed86f
A	audit/reconstruction/github_blobs/4c874bd4edb77e7129a965e749fc7c2479e2af91
A	audit/reconstruction/github_blobs/4d1108f9e7cc28fb0e5c748ff5c5f36c5ed1c7c9
A	audit/reconstruction/github_blobs/504816b80ced1f6efc9339b04aed8c18dbb92952
A	audit/reconstruction/github_blobs/50d92ec3fce85a39fa1cced31001da9d0d0e55ec
A	audit/reconstruction/github_blobs/58268be46a2ad303a0eb1ca3052911733aefaff2
A	audit/reconstruction/github_blobs/5921e8c62eefad2030f2a8b7e32e692bf3c95d1b
A	audit/reconstruction/github_blobs/5930accb52385b1d89547841f57d5ebd640aa1ac
A	audit/reconstruction/github_blobs/59f0cc7df1e70709c8f785b46bf2f4130c0403a8
A	audit/reconstruction/github_blobs/5a286612376e338bfd8e3d57e9c7e05a91d7202c
A	audit/reconstruction/github_blobs/5ac746bf2f2486fd71caff706a3c084556b8779f
A	audit/reconstruction/github_blobs/5d43a57a06ed1afcb311e7a60285e81f787bd3d2
A	audit/reconstruction/github_blobs/5da1fcf2fdb693e609d0c0bca92e2e2badb92515
A	audit/reconstruction/github_blobs/5dd638393fcb4d26d660516ba796c055360004a4
A	audit/reconstruction/github_blobs/5fc47e3fa7cb83931135524c955b21f2c1a308e7
A	audit/reconstruction/github_blobs/62d62f8ed8e7aa9916fd37650cd752e09a1af35c
A	audit/reconstruction/github_blobs/63d3b9e9509d0e90922945e445247bf1bf809424
A	audit/reconstruction/github_blobs/64d33adf27ed300c9192500195c9f40e5486397b
A	audit/reconstruction/github_blobs/66a63d8cd9de12b6fca2467262e223dbead652d8
A	audit/reconstruction/github_blobs/677c879e09f2cefa11fec23fe5f6d1021dd43f9b
A	audit/reconstruction/github_blobs/67a7f8ddf9129180d5a3ece661f93cc188ffce6b
A	audit/reconstruction/github_blobs/688f4a89f2eda6fe1b428224194152c9ac729662
A	audit/reconstruction/github_blobs/6b44c3019a07b06331fcd8feadbd46fc76e2b819
A	audit/reconstruction/github_blobs/6b987fa166b91cb53f1a2c1b45e97bb59ff4fa32
A	audit/reconstruction/github_blobs/6bbc20086fa0c0c28ed2457631e2fc149160980a
A	audit/reconstruction/github_blobs/6d5f90e3f6d7408219595340c1de8626fbd52c51
A	audit/reconstruction/github_blobs/70bc689b5312b783e8d43b9c113d6b506bd1312e
A	audit/reconstruction/github_blobs/74ee033e42eb1f3dd7acd551c55f1fb91f4ab0f6
A	audit/reconstruction/github_blobs/76a918faa617736e5d114a5069b31d378a6c8cfa
A	audit/reconstruction/github_blobs/774b2291ebd2cbb8cfaa4a56c58e6143a3eea4ba
A	audit/reconstruction/github_blobs/799c5dcb14217df89d92ab59b16b7b901ab3d375
A	audit/reconstruction/github_blobs/7cf9800ed31474e0cb4ef191ae1b9b2effcd7b38
A	audit/reconstruction/github_blobs/7d6347ea8280fddfa6047f03cf30f071c8a896e2
A	audit/reconstruction/github_blobs/7e7984273195d168e492ad0cb04c3bd559c7312b
A	audit/reconstruction/github_blobs/85c582b76700d90f7169c9b186d7f388372a3625
A	audit/reconstruction/github_blobs/869be6e90555014fb1f1d34ddd4f8943a8afbb97
A	audit/reconstruction/github_blobs/87c5e8f14b5062d6f4c25bd3f7a9dc8e526a5596
A	audit/reconstruction/github_blobs/8824c04cf6acda2baea041825845f38516320b08
A	audit/reconstruction/github_blobs/88af5f46ce0b7f48a12f7c33789f27ed5fb1b707
A	audit/reconstruction/github_blobs/8aee0cda75ebd20059a4a67aa7cd9e6d5cf0d83a
A	audit/reconstruction/github_blobs/8e5d02cac71110a8d3b01757df8e5fde5c96b733
A	audit/reconstruction/github_blobs/90dfe0970111ddffd0758e783b3ddfe91ecdc1f9
A	audit/reconstruction/github_blobs/94534c0f6ebf9a21aed338e016f6b016b2613715
A	audit/reconstruction/github_blobs/9463997c1a3a60922599144f380ec07a69aa7852
A	audit/reconstruction/github_blobs/9488ca77d348f5af86e35368bc38d067abad83fe
A	audit/reconstruction/github_blobs/9667bb243831c44dd379cc7460900c634a587431
A	audit/reconstruction/github_blobs/971de4b79ff1bdba36f2574098b976583bca064b
A	audit/reconstruction/github_blobs/994cb92dab37a947b2fcb329fd66e79c34229643
A	audit/reconstruction/github_blobs/9991c9fd326b772930f1bc361ef0caea6dca32ad
A	audit/reconstruction/github_blobs/9c7b8137e2c3260710592444f923af399eb49414
A	audit/reconstruction/github_blobs/9ccd377cebb98751e0322c5acdf090291e477005
A	audit/reconstruction/github_blobs/9d067a5b632a631fcf880e312393d6227ec7ade1
A	audit/reconstruction/github_blobs/9d5c4dc0d1b2251673d3082a60fb611b5aea379b
A	audit/reconstruction/github_blobs/9fe8c488bc6400c88ab0ae703c8364d77b642a1a
A	audit/reconstruction/github_blobs/a7f2afbcc44349ca4971bc0ba6f26c827a9cae7d
A	audit/reconstruction/github_blobs/a90f0bd3ae4b6f625347b824a56ab5ce66297168
A	audit/reconstruction/github_blobs/aa72e755891a0f70d71d52040b96854af8d1c10a
A	audit/reconstruction/github_blobs/aaeca2735208e0b9712bb01dc1c59c9dad989edb
A	audit/reconstruction/github_blobs/aba68e23fbc03645d01fd1a3c446cf9dd5ecce5b
A	audit/reconstruction/github_blobs/ac609ec691d78778575b6d1e5ad789b7e53056f4
A	audit/reconstruction/github_blobs/add98d50057ae8f0804461ecfcb58f9ee27c0669
A	audit/reconstruction/github_blobs/b08eb94ff47933712c3877cbcec64ab9afdcf1d4
A	audit/reconstruction/github_blobs/b1a1a88b9cddc44b76888f2d307f0e219141d624
A	audit/reconstruction/github_blobs/b290d8dcb71c58bb50c01aaecda87337c6285c71
A	audit/reconstruction/github_blobs/b4df641840be11dac9634f26c8bc451488a4feb6
A	audit/reconstruction/github_blobs/b5207a2ddc55f8b4ebd36ca353f89791a424ce39
A	audit/reconstruction/github_blobs/b5edd95fc7711dde55753105dae84b63b19a09b5
A	audit/reconstruction/github_blobs/b7a93bd69038b8239d306063085d96321a624102
A	audit/reconstruction/github_blobs/b9042c39cc6bbcb2ccb10ce972742afc29b40f96
A	audit/reconstruction/github_blobs/bc4497fafd20ae739b6ddbf5f28d9564086e70e1
A	audit/reconstruction/github_blobs/bfc9cc1ab79af709ae50c793ab13dc96a6589140
A	audit/reconstruction/github_blobs/bff6a0d70ff1119fdb207c1fc2d420ee135459d4
A	audit/reconstruction/github_blobs/c471f10135a1ba086b3439a760fe67726baa0894
A	audit/reconstruction/github_blobs/c4739f413c52a9911065afb005a7dfc841e305ce
A	audit/reconstruction/github_blobs/c4f2a0c5ac154ad7008843930c556db3d44d388f
A	audit/reconstruction/github_blobs/c505f90b681c43985f139b84ba7c3e80335741f9
A	audit/reconstruction/github_blobs/c6ce16a17a82e0095a347d2698a123c2a8cfb4e8
A	audit/reconstruction/github_blobs/c830f2044447add262133d9eceb927e71b6885e2
A	audit/reconstruction/github_blobs/c8bd00ca4aefb2c0cd375d9d2c1db42844635aaf
A	audit/reconstruction/github_blobs/c91d454c0eac74c843a382d1e2bf3a2d2d60e876
A	audit/reconstruction/github_blobs/ca93eb54835e035685f230c71a736e349db63e14
A	audit/reconstruction/github_blobs/cd53c132fa8ccdc3c91aeff7280ad9b3497a9615
A	audit/reconstruction/github_blobs/d236a6590a33b009ad13deb66bd795576a0dc4db
A	audit/reconstruction/github_blobs/d4082b7afc5e10d43fab8f9c23950abf351b1f2a
A	audit/reconstruction/github_blobs/d5a55f6ca39ac42e35e2e918d55f2f3c5443ff43
A	audit/reconstruction/github_blobs/d7c94f62657ed31fddcce08fc2b9639c1ca89324
A	audit/reconstruction/github_blobs/d99305a19ef20ce0b2e94f157536abb41843fb52
A	audit/reconstruction/github_blobs/d9a9e66ad0780bebd211c958602b9e3df0a51557
A	audit/reconstruction/github_blobs/dec5e5aaa35966a90302db8dc1f68fd55a3851c7
A	audit/reconstruction/github_blobs/e437660adb28e428a4a404febebfb1135e45a910
A	audit/reconstruction/github_blobs/e51ea82bb7dfc4c64cefa49948803188386529f0
A	audit/reconstruction/github_blobs/e6e13b045654394f9a607e41e62a62bf632c9be9
A	audit/reconstruction/github_blobs/e70adac30a31b4f209747438cbfcbd43fa58c527
A	audit/reconstruction/github_blobs/ebe049a1d549cc2c5dc554c083c35c14742d1e6a
A	audit/reconstruction/github_blobs/ec97585abbdad3e37c6665a7214292ac8b6ad17e
A	audit/reconstruction/github_blobs/ece299469be325d66a240761e989b0976ade40f5
A	audit/reconstruction/github_blobs/edf8f86bf3344d48d410ee981d69f3a1fb995f67
A	audit/reconstruction/github_blobs/ef15eb2be5554892b7c6a518c084730a1de289df
A	audit/reconstruction/github_blobs/ef7878e9d7a9f88ac4cb9958cd1e7f4724501f33
A	audit/reconstruction/github_blobs/f15c399fc64878f2a17dbb966b9310da8d051492
A	audit/reconstruction/github_blobs/f2d3935f5932077c75008da61a24a6b07e70ab2d
A	audit/reconstruction/github_blobs/f423f12271d93e81a8d7da7946b9cf2139a29a9b
A	audit/reconstruction/github_blobs/f5bf3986664c1509f35ad6d728a77efba2503739
A	audit/reconstruction/github_blobs/f625c6f45b27126e0acaf1e4b7d381b3d3c1e896
A	audit/reconstruction/github_blobs/fa27794a613d82145c3943130a70b63367fcbcbd
A	audit/reconstruction/github_blobs/fb197a922bb4ed27b5e8d5fb04f1caddaeb27878
A	audit/reconstruction/github_blobs/fb661d84aa0fb33e9b684d6cc33afa2ab8c5de71
A	audit/reconstruction/github_blobs/fe84a32d15b26b32994dc1c28a31e0a77e4ba91f
A	audit/reconstruction/github_bn_commits.json
A	audit/reconstruction/github_bn_tree.json
A	audit/reconstruction/github_commits.json
A	audit/reconstruction/github_da_branches.json
A	audit/reconstruction/github_evidence_manifest.json
A	audit/reconstruction/github_origin_branches.json
A	audit/reconstruction/github_phuong_commits.json
A	audit/reconstruction/github_phuong_tree.json
A	audit/reconstruction/github_source/aggregation.ipynb
A	audit/reconstruction/github_source/aggregation_cells.txt
A	audit/reconstruction/github_source/src__commons__create_train_test.py
A	audit/reconstruction/github_source/src__datasets.py
A	audit/reconstruction/github_source/src__modeling_thuy__classification_main.py
A	audit/reconstruction/github_source/src__modeling_thuy__classification_train.py
A	audit/reconstruction/github_source/src__modeling_thuy__constants.py
A	audit/reconstruction/github_source/src__modeling_thuy__data_loader.py
A	audit/reconstruction/github_source/src__modeling_thuy__models_folder__model_census_kdd.py
A	audit/reconstruction/github_source/src__modeling_thuy__output__best_result_from_csv.ipynb
A	audit/reconstruction/github_tree.json
A	audit/reconstruction/make_report.py
A	audit/reconstruction/math328 project proposal_pdf.txt
A	audit/reconstruction/post_processing_cells.txt
A	audit/reconstruction/reconstruct_analysis.py
A	audit/reconstruction/reconstructed_final_results.csv
A	audit/reconstruction/report_matches.csv
A	audit/reconstruction/src__commons__create_train_test.py.json
A	audit/reconstruction/src__datasets.py.json
A	audit/reconstruction/src__modeling_thuy__classification_main.py.json
A	audit/reconstruction/src__modeling_thuy__classification_train.py.json
A	audit/reconstruction/src__modeling_thuy__constants.py.json
A	audit/reconstruction/src__modeling_thuy__data_loader.py.json
A	audit/reconstruction/src__modeling_thuy__models_folder__model_census_kdd.py.json
A	audit/reconstruction/src__modeling_thuy__output__best_result_from_csv.ipynb.json
A	audit/reconstruction/statistics_check.txt
A	audit/reconstruction/test_f1_macro_end_all_dataset_means.csv
A	audit/reconstruction/test_f1_macro_end_no_gauss_dataset_means.csv
A	audit/reconstruction/test_f1_macro_max_all_dataset_means.csv
A	audit/reconstruction/test_f1_macro_max_no_gauss_dataset_means.csv
A	audit/reconstruction/trace_checks.py
A	audit/reconstruction/trace_checks.txt
A	audit/reconstruction/traced_run_paths.json
A	audit/reproduce.py
A	audit/results_review/PAPER_COMPARISON.md
A	audit/results_review/RESULTS_REVIEW.md
A	audit/results_review/analysis.json
A	audit/results_review/analysis_output.txt
A	audit/results_review/analyze.py
A	audit/results_review/archive_manifest.json
A	audit/results_review/build_comparison.py
A	audit/results_review/drive/107D2dLcVBlknbsNSFN6yw42Wqo04MR4s.json
A	audit/results_review/drive/15eDjTcRVcsAgAMeQUjoDt7Q1OkGI1W1Y.json
A	audit/results_review/drive/17AdUQkXb9Jr0kKN6og9NfXV5qS6uwKQ4.json
A	audit/results_review/drive/18L3VF2Hfab3b5Xgdd87qEaAnp0X1fhzG.json
A	audit/results_review/drive/196WozcMkIR6bZET4lu0LI5QWS_q2nKU1.json
A	audit/results_review/drive/1COKnOxUlPOO2_VCy8ItMUhuCWiucY73S.json
A	audit/results_review/drive/1FDGcbQCR6jBwoey2iHu7IOpX056wLdpK.json
A	audit/results_review/drive/1FqPwSMx8C9e-iQhonYhb6hIcKbFh82aG.json
A	audit/results_review/drive/1GnYTdAjuhlW_2e9s6JyeM9q0XUHhu2nV.json
A	audit/results_review/drive/1HJw5bN1sE3U-ws-Hpu97m5FXxgwJlY65.json
A	audit/results_review/drive/1I87Rpgt_ro5rLajwXH0oS8O3BD2bteZW.json
A	audit/results_review/drive/1Ic2S7_5cvE7A0M3zBtinNMqQnubGWqMw.json
A	audit/results_review/drive/1JIzMWt42hemUYcDbEYzQWe5CgPxM-542.json
A	audit/results_review/drive/1LbwAglbRrSvlBEomeOiAYhrsCiFAxdIx.json
A	audit/results_review/drive/1NGFuyRVRSxubA_LVzhlzee3Y9W1i1Bnh.json
A	audit/results_review/drive/1S4KDDgbN5QZbnCtNiZJR3y0Jp7RHIRWm.json
A	audit/results_review/drive/1Vobr0_G65x6p8J80LAdwurphQiga2KQb.json
A	audit/results_review/drive/1WQiJuH6X2cGVlIGFGHYdl1JNXKSCkHBx.json
A	audit/results_review/drive/1_HFUiVO69L-symSqTZX1sErVAsORipiC.json
A	audit/results_review/drive/1biOfN8us1SZE1FQgBGDF5UVmnNlN49BR.json
A	audit/results_review/drive/1kmXKrzcvTVLzwD5_ZySjKYCYGZ8LOeRF.json
A	audit/results_review/drive/1l1Dp6yDA7-OeM5bnA1LF-qm0Op5qrDSH.json
A	audit/results_review/drive/1lY5lnMajCZ_MU09mh3u98IjkiNvBVkiC.json
A	audit/results_review/drive/1m2ajH-MfeiJr4omWZzhYWcrWG8jnfO7m.json
A	audit/results_review/drive/1qCzp8WPPjk4wh9kxOaqDZQlswpD0KHY6.json
A	audit/results_review/drive/1sgkvQjAHTv34FHHlkndWHwsdgr3gMHvD.json
A	audit/results_review/drive/1zmeYpVvcR896da3_sU2fhVvuDxiM2S66.json
A	audit/results_review/drive/best_result_from_csv.ipynb
A	audit/results_review/drive/final_results.csv
A	audit/results_review/drive/formatted_results.csv
A	audit/results_review/drive/listings_level2.jsonl
A	audit/results_review/drive/listings_level3.jsonl
A	audit/results_review/drive/listings_level4.jsonl
A	audit/results_review/drive/log_manifest.json
A	audit/results_review/drive/logs/11QhtQiGlqDAV7d4lt8seSsvSHnh9q6rF.csv
A	audit/results_review/drive/logs/11YwvyTdO35rc424033rmFyoUv6_Om7vk.csv
A	audit/results_review/drive/logs/13Y42i6uknDzXGhNiMbNyxTwTpEgi0UjM.csv
A	audit/results_review/drive/logs/14j9pevCSz5i9wMDV4h6eObKsDrZIRiDo.csv
A	audit/results_review/drive/logs/15Ck0JW4Mn42JSCGSdCwF0E04eQu49ZZ8.csv
A	audit/results_review/drive/logs/173KnSGXGIZQ9ddvBl9HKmMJH4L_YGyuj.csv
A	audit/results_review/drive/logs/17GdBkTXqPnsYj0-u9u-bK3L3VZgyXepn.csv
A	audit/results_review/drive/logs/1BlxpouAO6JJNtApUUGDogjUuoJNSIncM.csv
A	audit/results_review/drive/logs/1Gayp3TXbtBkvSHg3if6EHlAcovHirMUB.csv
A	audit/results_review/drive/logs/1Gcu9L1f2RxPe4sBb0tDzVdS1gD_mekFt.csv
A	audit/results_review/drive/logs/1HMyk_eaqO00gSVIxlHlINx57Az5cyDN3.csv
A	audit/results_review/drive/logs/1JOE-IXRwmDv26DN4heZJoldrxJFomDio.csv
A	audit/results_review/drive/logs/1KFrAktILB4YiLW2zUme_ivy19XZdstTv.csv
A	audit/results_review/drive/logs/1K_PPLAaXHJ6UqZjFosHpQUI0IHzqgtgH.csv
A	audit/results_review/drive/logs/1KzCUpoQU-Vnq1hdN_KxK0Krug2UzfoMc.csv
A	audit/results_review/drive/logs/1L9Q-1rsSLG6udB-CrIbMPE34T3u5eaai.csv
A	audit/results_review/drive/logs/1NMo-gdTFeDe-6mQu0aA9PWuPidVzJOsO.csv
A	audit/results_review/drive/logs/1NdUXhLsKRG3mWaCuEzdIn3XDEcMzMD5P.csv
A	audit/results_review/drive/logs/1O-cZMBY85LNMrjGT_zI8mhR_I7DYxJmP.csv
A	audit/results_review/drive/logs/1OAzyM0TmTYl4hWPNfwLu542UV6gfkHc8.csv
A	audit/results_review/drive/logs/1P8dy9MK61nQzjAWPIMfGE_QM2Z1JxAd4.csv
A	audit/results_review/drive/logs/1PIAN5SKqVZlAyATg1Qf1KGzxiiBcqwXC.csv
A	audit/results_review/drive/logs/1QSz9VvLFgxxI6lCpiTXyuN5iVdpScLdi.csv
A	audit/results_review/drive/logs/1QbLsMksUANhqdrDWzGb6yTJr-QTGZDUq.csv
A	audit/results_review/drive/logs/1SHuLvmxoFpSZsT8tGwbp5XU1n9GCYO2I.csv
A	audit/results_review/drive/logs/1T-iIEQHYc1gRLFDW4OUIpJIoDIrEgX1x.csv
A	audit/results_review/drive/logs/1TIK7fi_FY8mD5xVOHsRg3tuynijTfjpu.csv
A	audit/results_review/drive/logs/1UChP8PGbY4PPM7nr4MSGbawRfCU_hvpR.csv
A	audit/results_review/drive/logs/1VV6qf2ExCgW52g0C9iSAdrpXvNbaT-zN.csv
A	audit/results_review/drive/logs/1Ws6facwk2MIhRKFk0T3u-cj7wbsefrJJ.csv
A	audit/results_review/drive/logs/1XH_tRcr-4ppkXh5y4jYZUoOPRY5MSMLk.csv
A	audit/results_review/drive/logs/1_inx1HAa8BoGWnx4_LouS46Z2NQBsME3.csv
A	audit/results_review/drive/logs/1a4aYW6WR-td97Rzje-qTzqcwHDyQihpt.csv
A	audit/results_review/drive/logs/1aQoAtXfp6osgRrlZ0jS51Jnpih76Ml8n.csv
A	audit/results_review/drive/logs/1b0tJvoV1q7COP0098JYxjPMdztCl9pjm.csv
A	audit/results_review/drive/logs/1c1pWvkkT1b8ghscD77LJQznIj2yvyOTs.csv
A	audit/results_review/drive/logs/1coHMQw-CrB33OTG3evOqDMHCEdqg4AD_.csv
A	audit/results_review/drive/logs/1e68-OXZ2flpnZYjJpJ-Tq0IgzybAUaM8.csv
A	audit/results_review/drive/logs/1edQaBgrM9_ehgr_gfRBI86oVMKoakfne.csv
A	audit/results_review/drive/logs/1eiKXhQD3ZUWTuIL6GUFIJCEba7EMULzZ.csv
A	audit/results_review/drive/logs/1f4a1WdssxGGAveDmxF_HM1LlUGSNvCld.csv
A	audit/results_review/drive/logs/1ftVUCKShZuCrVw7bnck14F9oFe0Zzkix.csv
A	audit/results_review/drive/logs/1gUFf1GxRThAKZtt1VhIdxVPZN817TM2h.csv
A	audit/results_review/drive/logs/1gx-riwD1xNCvpr_4eofFoM9m8cpaoxaf.csv
A	audit/results_review/drive/logs/1h0kOe54IzK5B42pzaUMbflITfUhsYZzU.csv
A	audit/results_review/drive/logs/1hlem4y810z8q7trbKiUHV_FC2n2aRfpQ.csv
A	audit/results_review/drive/logs/1iSuHWbX5TKla7H7cEU5fndluWuB3WS6g.csv
A	audit/results_review/drive/logs/1iu6uTOjrRl1GZYrLDWObqczIdO3aRXbw.csv
A	audit/results_review/drive/logs/1kX3Rn8EN7nVWI-povLCtlI8JNaavxRqY.csv
A	audit/results_review/drive/logs/1kXJams8mTa5YnPEwT1FeNh6EUrKp6-49.csv
A	audit/results_review/drive/logs/1kbyfcc-uwcis1NmLkmBxojDIVcaXoy4g.csv
A	audit/results_review/drive/logs/1kr4vbknGwrVYa3oMfM5uXm_MYGDxHJxl.csv
A	audit/results_review/drive/logs/1lLr7xCQJ4iRSb4grd5utTjZ_6c-EhG1t.csv
A	audit/results_review/drive/logs/1m2DIlPJqr4MWu0_3ifEAXzcOhQ1ri69_.csv
A	audit/results_review/drive/logs/1mBu2EWK-8Z-gvIqdEBV7_f5rbe6kOsZI.csv
A	audit/results_review/drive/logs/1mVyX_qvyfRRdMSlxX63SzwU3SypFxWj7.csv
A	audit/results_review/drive/logs/1nPp95mw2nXa7Bsaf-Dk_oMCqDnSKe1JO.csv
A	audit/results_review/drive/logs/1n_hiqP1fqxzqvGsHxliNayvHaumZZbHY.csv
A	audit/results_review/drive/logs/1ntzVxx9sWRn5a4NKpSO4svqWhu-sso6g.csv
A	audit/results_review/drive/logs/1p20PuX8YfxCGpHYJlb2m_c75LVWab09z.csv
A	audit/results_review/drive/logs/1p4CKJJOtbM1O0BkMkCuj20kv17EoJWOx.csv
A	audit/results_review/drive/logs/1pJwPykGKFdqPxKNh7F_U7SKl-4aAq360.csv
A	audit/results_review/drive/logs/1snauLB2QPD27LnBJolhdiNmr-b-EQaYP.csv
A	audit/results_review/drive/logs/1sriacJVVi7BA-8B2St_K5rg81P4ChpwN.csv
A	audit/results_review/drive/logs/1tUtUikUOm3gDxx1AAENVjZzBzmjxZHyQ.csv
A	audit/results_review/drive/logs/1tYfKzc1BW5FwBHEALaBcpwZzbLSo5zQd.csv
A	audit/results_review/drive/logs/1thPD5M46AqVuFArFzPo7pYuEJ-kkWJ1v.csv
A	audit/results_review/drive/logs/1ttKgK2qdm7jKYdQYP2ENbgqQ2u2LK886.csv
A	audit/results_review/drive/logs/1vx-qIg0U0VgXgrRkJoove6m74hl1Dazw.csv
A	audit/results_review/drive/logs/1wVqYFcNj2pWREQPZLJ2RC_Zqmj-YQa21.csv
A	audit/results_review/drive/logs/1xTU93bKZoSrNMoRMyjuEFNkwRrC9TO9W.csv
A	audit/results_review/drive/logs/1yq2eyFp1SHkOOOOwg8q9i4b8SKyQEZuH.csv
A	audit/results_review/drive_read.py
A	audit/results_review/inspect.mjs
A	audit/results_review/local_download.py
A	audit/results_review/local_inventory.txt
A	audit/results_review/local_log_manifest.json
A	audit/results_review/local_logs/023da30841c4b4f616bf.csv
A	audit/results_review/local_logs/02980ce99eac444c6036.csv
A	audit/results_review/local_logs/029c6ef06b810724d6ff.csv
A	audit/results_review/local_logs/033682e5d80ac872fb93.csv
A	audit/results_review/local_logs/05edf264c7b119c6e2e2.csv
A	audit/results_review/local_logs/05fa27365ad3a419761b.csv
A	audit/results_review/local_logs/0724d682f36f85846835.csv
A	audit/results_review/local_logs/0cae5704426921383bfa.csv
A	audit/results_review/local_logs/0d2f997d730e286a6f9f.csv
A	audit/results_review/local_logs/0eb2881a00ddeda9136a.csv
A	audit/results_review/local_logs/1086dbf3bcc24ee8bae0.csv
A	audit/results_review/local_logs/10af490f998f1ec76945.csv
A	audit/results_review/local_logs/12479a847b1b0a30b25f.csv
A	audit/results_review/local_logs/130b95938cbebcfc6d08.csv
A	audit/results_review/local_logs/13ee9c99941d2e810977.csv
A	audit/results_review/local_logs/1f02ffdff465a5b22dc7.csv
A	audit/results_review/local_logs/308fc5b8ca03a72d91b7.csv
A	audit/results_review/local_logs/3223e53cf7f40126a48f.csv
A	audit/results_review/local_logs/3640ab41592aa1eb6233.csv
A	audit/results_review/local_logs/386bc01699eff19dd6b4.csv
A	audit/results_review/local_logs/394b6c77a4ea3aaec977.csv
A	audit/results_review/local_logs/3cd71cfa3c1c700569e3.csv
A	audit/results_review/local_logs/3cece1b106207418dfd9.csv
A	audit/results_review/local_logs/401964fdb92c33f178e0.csv
A	audit/results_review/local_logs/481c616610ed850d84b4.csv
A	audit/results_review/local_logs/4ac276167474f0666bb2.csv
A	audit/results_review/local_logs/4b0e67f94875af5088ee.csv
A	audit/results_review/local_logs/4b7615c89e41bfcff2ae.csv
A	audit/results_review/local_logs/4d106200a72870a6e9f8.csv
A	audit/results_review/local_logs/525e3c55a7b859945b52.csv
A	audit/results_review/local_logs/5400afda1a60ca450aac.csv
A	audit/results_review/local_logs/540ec030a57a49e517bb.csv
A	audit/results_review/local_logs/581c0f05df0821088ab4.csv
A	audit/results_review/local_logs/5e36c3f77c135cdefe23.csv
A	audit/results_review/local_logs/5ffe09730381ac089f1f.csv
A	audit/results_review/local_logs/603be2bd5677cb3d1c6c.csv
A	audit/results_review/local_logs/6180f02c1b94fa717d55.csv
A	audit/results_review/local_logs/66d1d823bc50d4dc1913.csv
A	audit/results_review/local_logs/695e1867d024114be7a0.csv
A	audit/results_review/local_logs/69890e09f3fd1cb3529b.csv
A	audit/results_review/local_logs/6a12eab6a1f437dd8a37.csv
A	audit/results_review/local_logs/70d5f79ca9bae81bd349.csv
A	audit/results_review/local_logs/74d6f4cd17625b8d65e6.csv
A	audit/results_review/local_logs/756bfe39c460af36adc2.csv
A	audit/results_review/local_logs/7d4b070aaa3759f3e0a4.csv
A	audit/results_review/local_logs/7dc2392339a72b5292cc.csv
A	audit/results_review/local_logs/83d6301824498438674c.csv
A	audit/results_review/local_logs/85faf7d92cfcaec21e6e.csv
A	audit/results_review/local_logs/867be875e848b642453b.csv
A	audit/results_review/local_logs/8880430b25041151611c.csv
A	audit/results_review/local_logs/88e806303c4d5cd857df.csv
A	audit/results_review/local_logs/8bcaab2ffa7e957d7630.csv
A	audit/results_review/local_logs/94e0cf059b454a093cb4.csv
A	audit/results_review/local_logs/a0548fa1ac13fa92e83a.csv
A	audit/results_review/local_logs/a131121ac0fc8dd01342.csv
A	audit/results_review/local_logs/a8224d7652baf7b1e96f.csv
A	audit/results_review/local_logs/ad50aabe0ae9ab5334b3.csv
A	audit/results_review/local_logs/ae482e9a8b549c005468.csv
A	audit/results_review/local_logs/b1a65b3bd6c250568f33.csv
A	audit/results_review/local_logs/b7407b892cabae3fa343.csv
A	audit/results_review/local_logs/b8593938ad04ab80fd35.csv
A	audit/results_review/local_logs/b9254b83c0741e624758.csv
A	audit/results_review/local_logs/bc76cd632b06ec2a3c2f.csv
A	audit/results_review/local_logs/bde0bb20e8ed04a45770.csv
A	audit/results_review/local_logs/c0f9fc85bdd184318f91.csv
A	audit/results_review/local_logs/c13a40414e37fbc90f5b.csv
A	audit/results_review/local_logs/c17a6eb9e74d884abe35.csv
A	audit/results_review/local_logs/c2c0db5481cb61c2e995.csv
A	audit/results_review/local_logs/c4401a72606ba50c6fe7.csv
A	audit/results_review/local_logs/c4787e94c5aaf4912929.csv
A	audit/results_review/local_logs/cf156ca876d46684b207.csv
A	audit/results_review/local_logs/d435276e8a2447605cc4.csv
A	audit/results_review/local_logs/dc17a220b8ecdecdf176.csv
A	audit/results_review/local_logs/dcebafddc00f1e1115a1.csv
A	audit/results_review/local_logs/dcf1f31fe012b95ba0e1.csv
A	audit/results_review/local_logs/dd9ee6d91a560269c2f4.csv
A	audit/results_review/local_logs/e1076d83fdb362a4c1c5.csv
A	audit/results_review/local_logs/e365adb5a686cf2173e2.csv
A	audit/results_review/local_logs/e6b0c5b241bd876f66d3.csv
A	audit/results_review/local_logs/e919151b5fa29306ce44.csv
A	audit/results_review/local_logs/ea5aecbdf02c99e0ca83.csv
A	audit/results_review/local_logs/f07e4d24d231efdf94a2.csv
A	audit/results_review/local_logs/f3bbb0bde14112eda354.csv
A	audit/results_review/local_logs/f58ce2dbd675116415a9.csv
A	audit/results_review/local_logs/fd77a7246dee28dc8ce9.csv
A	audit/results_review/local_overlap.json
A	audit/results_review/local_overlap.py
A	audit/results_review/local_overlap_output.txt
A	audit/results_review/reconstruct.py
A	audit/results_review/reconstructed.json
A	audit/results_review/reconstruction_output.txt
A	audit/results_review/sensitivity.json
A	audit/results_review/sensitivity.py
A	audit/results_review/sensitivity_output.txt
A	audit/results_review/versions.json
A	audit/results_review/versions.py
A	audit/results_review/versions_output.txt
A	audit/results_review/workbook.json
A	audit/results_review/workbook_comparison.py
A	data/mnist12/old_split/ensemble_results.txt
A	data/mnist28/old_split/ensemble_results.txt
A	flowchart.png
A	flowchart1.png
A	flowchart_final.png
M	pyglrm
A	sdv trained model/mnist28/old_split/mnist28_synthesizers_all.zip
M	src/commons/__pycache__/create_train_test.cpython-311.pyc
A	src/modeling_thuy/post_processing.ipynb
M	src/synthesize_data/create_synthetic_data/CreateSyntheticData.py
M	src/synthesize_data/create_synthetic_data/EDA_Credit.ipynb
M	src/synthesize_data/create_synthetic_data/__pycache__/CreateSyntheticData.cpython-311.pyc
M	src/synthesize_data/create_synthetic_data/__pycache__/adult.cpython-311.pyc
M	src/synthesize_data/create_synthetic_data/__pycache__/census.cpython-311.pyc
M	src/synthesize_data/create_synthetic_data/__pycache__/census_kdd.cpython-311.pyc
M	src/synthesize_data/create_synthetic_data/__pycache__/covertype.cpython-311.pyc
M	src/synthesize_data/create_synthetic_data/__pycache__/credit.cpython-311.pyc
M	src/synthesize_data/create_synthetic_data/__pycache__/intrusion.cpython-311.pyc
M	src/synthesize_data/create_synthetic_data/__pycache__/mnist12.cpython-311.pyc
M	src/synthesize_data/create_synthetic_data/__pycache__/mnist28.cpython-311.pyc
M	src/synthesize_data/create_synthetic_data/__pycache__/news.cpython-311.pyc
M	src/synthesize_data/synthesizer.py
A	tmp/abstract_review/poster.png
A	tmp/poster_review/embedded_0.png
A	tmp/poster_review/embedded_1.png
A	tmp/poster_review/poster-1.png
A	tmp/poster_review/rplot-38.png
A	LOCAL_CHANGES_BEFORE_REFACTOR.md
```

## Nested repository changes

### GLRM

Checkpoint commit: `f6b7482fe4b70f8cebad02d9843f49df456bba49`.

```text
M	glrm/__init__.py
M	glrm/convergence.py
M	glrm/glrm.py
A	glrm/glrm_df.py
M	glrm/loss.py
M	glrm/reg.py
```

### pyglrm

Checkpoint commit: `6c29ee5960f5e5a388f3433fca3685b4673ad6aa`.

```text
M	.gitignore
```

## Ignored archive and SDGym files remaining on disk

These files are outside the Git checkpoint.

| Path | Bytes |
|---|---:|
| data/mnist12/old_split/mnist12_test.csv | 4060472 |
| data/mnist12/old_split/mnist12_train.csv | 16240472 |
| data/mnist12/old_split/onehot_mnist12_sdv_100k.csv | 29000472 |
| data/mnist12/old_split/onehot_mnist12_sdv_categorical_100k.csv | 29000472 |
| data/mnist12/old_split/onehot_mnist12_sdv_gaussian_100k.csv | 29000472 |
| data/mnist12/old_split/onehot_mnist12_sdv_pca_gmm_cat_100k.csv | 29100473 |
| data/mnist12/old_split/onehot_mnist12_sdv_pca_gmm_num_100k.csv | 29100473 |
| data/mnist12/old_split/onehot_mnist12_sdv_rf_100k.csv | 29000472 |
| data/mnist12/old_split/onehot_mnist12_sdv_tvae_100k.csv | 29000472 |
| data/mnist12/old_split/onehot_mnist12_sdv_xgb_100k.csv | 29000472 |
| data/mnist12/old_split/onehot_mnist12_test.csv | 4060472 |
| data/mnist12/old_split/onehot_mnist12_train.csv | 16240472 |
| data/mnist28/old_split/mnist28_test.csv | 21983032 |
| data/mnist28/old_split/mnist28_train.csv | 87923032 |
| data/mnist28/old_split/onehot_mnist28_sdv_100k.csv | 157003032 |
| data/mnist28/old_split/onehot_mnist28_sdv_categorical_100k.csv | 157003032 |
| data/mnist28/old_split/onehot_mnist28_sdv_gaussian_100k.csv | 157003032 |
| data/mnist28/old_split/onehot_mnist28_sdv_pca_gmm_100k.csv | 157103033 |
| data/mnist28/old_split/onehot_mnist28_sdv_pca_gmm_cat_100k.csv | 157103033 |
| data/mnist28/old_split/onehot_mnist28_sdv_rf_100k.csv | 601008846 |
| data/mnist28/old_split/onehot_mnist28_sdv_tvae_100k.csv | 601008846 |
| data/mnist28/old_split/onehot_mnist28_sdv_xgb_100k.csv | 601008846 |
| data/mnist28/old_split/onehot_mnist28_test.csv | 21983032 |
| data/mnist28/old_split/onehot_mnist28_train.csv | 87923032 |
| sdv trained model/mnist28/old_split/mnist28_TVAE_synthesizer.pkl | 8271096 |
| sdv trained model/mnist28/old_split/mnist28_synthesizer.pkl | 375028670 |
| sdv trained model/mnist28/old_split/mnist28_synthesizer_onlyX.pkl | 374354497 |
| SDGym/datasets/adult/adult.csv | 3518605 |
| SDGym/datasets/alarm/alarm.csv | 4174494 |
| SDGym/datasets/census/census.csv | 140900753 |
| SDGym/datasets/child/child.csv | 2304075 |
| SDGym/datasets/covtype/covtype.csv | 75170062 |
| SDGym/datasets/expedia_hotel_logs/expedia_hotel_logs.csv | 112685 |
| SDGym/datasets/insurance/insurance.csv | 3927307 |
| SDGym/datasets/intrusion/intrusion.csv | 50164050 |
| SDGym/datasets/news/news.csv | 6192526 |

## Secret cleanup

The checkpoint commits were reconstructed to omit 28 raw Drive HTML snapshots and redact eight signed download URL tokens. A final cleanup commit adds ignore rules, prevents raw HTML persistence, configures secret scanning, and records the scan scope in audit/SECRET_SCAN.md.
