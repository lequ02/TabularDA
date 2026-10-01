"""Read-only audit of a completed Linux run; write diagnostics beside this script."""
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "SDGym-research"))
from synthetic_data_benchmark import simulated_benchmark as bench
from synthetic_data_benchmark import run_simulated_methods as runner
RUN = ROOT / "SDGym-research/data/simulated_paper"
OUT = Path(__file__).with_suffix(".json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_prob(data, name):
    p = ROOT / f"SDGym/datasets/{name}/metadata_v0.json"
    network = json.loads(p.read_text())["tables"][name]["structure"]
    probability = np.ones(len(data))
    for i, state in enumerate(network["states"]):
        distribution = state["distribution"]
        if distribution["name"] == "DiscreteDistribution":
            probability *= data[state["name"]].map(distribution["parameters"][0]).to_numpy(float)
        else:
            variables = [network["states"][parent]["name"] for parent in network["structure"][i]] + [state["name"]]
            lookup = {tuple(row[:-1]): float(row[-1]) for row in distribution["table"]}
            probability *= np.array([lookup[row] for row in data[variables].itertuples(index=False, name=None)])
    return probability


def main():
    logging.getLogger("pgmpy").setLevel(logging.ERROR)
    rows = list(csv.DictReader((RUN / "simulated_methods_per_run.csv").open()))
    report = {"rows": len(rows), "seeds": sorted({r["seed"] for r in rows}),
              "datasets": sorted({r["dataset"] for r in rows}), "issues": [], "paper_scores": [],
              "utility_rechecks": [], "max_l_syn_error": 0., "hashes_checked": 0}
    manifests, models, tables = {}, {}, {}
    for number, r in enumerate(rows):
        dataset, seed, method = r["dataset"], int(r["seed"]), r["method"]
        folder = RUN / f"seed_{seed}" / dataset
        manifest = manifests.setdefault((dataset,seed), json.loads((folder/"manifest.json").read_text()))
        suffix = "result" if r["benchmark"]=="paper" else "labeled"
        record = json.loads((folder/f"{method}_{suffix}.json").read_text())
        if (record["dataset"],record["seed"],record["method"]) != (dataset,seed,method):
            report["issues"].append([dataset,seed,method,"identity mismatch"])
        for key in ("synthetic", "source"):
            if record.get(key+"_path"):
                report["hashes_checked"] += 1
                if sha(record[key+"_path"]) != record[key+"_sha256"]:
                    report["issues"].append([dataset,seed,method,key+" hash mismatch"])
        for split in ("train", "test"):
            if (dataset,seed,split) not in tables:
                p=folder/f"{split}.csv"
                if sha(p)!=manifest[split+"_sha256"]:
                    report["issues"].append([dataset,seed,split,"split hash mismatch"])
                tables[(dataset,seed,split)] = bench.read_table(p,dataset)
            if record.get(split+"_sha256") != manifest[split+"_sha256"]:
                report["issues"].append([dataset,seed,method,split+" record mismatch"])
        if sha(manifest["oracle_path"]) != manifest["oracle_sha256"]:
            report["issues"].append([dataset,seed,method,"oracle hash mismatch"])
        synthetic = bench.read_table(record["synthetic_path"],dataset)
        test = tables[(dataset,seed,"test")]
        if dataset in bench.MIXTURES:
            features=synthetic[["feature_0","feature_1"]]
            oracle=json.loads(Path(manifest["oracle_path"]).read_text())
            logp=bench.mixture_log_prob(features,oracle)
        else:
            if dataset not in models:
                models[dataset]=bench.bn_model(dataset)
            model=models[dataset]
            logp=bench.bn_log_prob(synthetic,model)
        error=abs(float(logp.mean())-record["l_syn"])
        report["max_l_syn_error"]=max(report["max_l_syn_error"],error)
        if error>1e-8:
            report["issues"].append([dataset,seed,method,"l_syn mismatch",error])
        if r["benchmark"]=="paper":
            entry={"dataset":dataset,"seed":seed,"method":method,"l_syn":record["l_syn"],"saved_l_test":record["l_test"]}
            if dataset in bench.MIXTURES:
                for n_init in (5,1):
                    gmm=GaussianMixture(n_components=len(oracle["means"]),covariance_type="diag",random_state=seed,n_init=n_init).fit(features)
                    entry[f"recomputed_l_test_n_init_{n_init}"]=float(gmm.score(test))
                recomputed=entry["recomputed_l_test_n_init_5"]
            else:
                from pgmpy.estimators import MaximumLikelihoodEstimator
                refit=bench.bn_model(dataset)
                refit.fit(synthetic,estimator=MaximumLikelihoodEstimator,state_names=manifest["categories"])
                recomputed=float(bench.bn_log_prob(test,refit).mean())
                entry["recomputed_l_test"]=recomputed
                # Count exact impossible rows separately from the numerical floor.
                probabilities=np.ones(len(synthetic))
                for cpd in model.get_cpds():
                    indices=[synthetic[v].map({s:i for i,s in enumerate(cpd.state_names[v])}).to_numpy(int) for v in cpd.variables]
                    probabilities*=cpd.values[tuple(indices)]
                entry["impossible_synthetic_fraction"]=float((probabilities==0).mean())
                meta=ROOT/f"SDGym/datasets/{dataset}/metadata_v0.json"
                if meta.exists():
                    historical=historical_prob(synthetic,dataset)
                    entry["historical_oracle_logp_max_error"]=float(np.max(np.abs(np.log(historical+1e-8)-logp)))
            if abs(recomputed-record["l_test"])>1e-8:
                report["issues"].append([dataset,seed,method,"l_test mismatch",recomputed-record["l_test"]])
            report["paper_scores"].append(entry)
        if seed==42 and dataset in ("grid","insurance") and method in (
                "ctgan-full","ctgan-rf","ctgan-xgb","ctgan-dnn","tvae-full","tvae-rf","tvae-xgb","tvae-dnn"):
            target=runner.target_name(dataset)
            labeled_test=runner.labeled(test,dataset)
            xfit,xtest,_=runner.encode_features(synthetic.drop(columns=target),labeled_test.drop(columns=target),labeled_test.drop(columns=target),dataset)
            predictions=RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1).fit(xfit,synthetic[target]).predict(xtest)
            acc=accuracy_score(labeled_test[target],predictions)
            f1=f1_score(labeled_test[target],predictions,average="macro",zero_division=0)
            entry={"dataset":dataset,"seed":seed,"method":method,"accuracy_error":abs(acc-record["test_accuracy"]),"f1_error":abs(f1-record["test_macro_f1"])}
            if max(entry["accuracy_error"],entry["f1_error"])>1e-10:
                report["issues"].append([dataset,seed,method,"utility mismatch",entry])
            report["utility_rechecks"].append(entry)
        if (number+1)%50==0:
            print(f"Audited {number+1}/{len(rows)} records",flush=True)
    OUT.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ("paper_scores","utility_rechecks")},indent=2),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
