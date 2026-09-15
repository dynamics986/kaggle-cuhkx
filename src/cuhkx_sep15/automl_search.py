"""AutoGluon HPO on leakage-safe clip summaries (not a sequence Transformer)."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from cuhkx_har.automl import LABEL, NUM_CLASSES, _import_predictor
from cuhkx_har.splits import fold_partition
from cuhkx_sep12.common import submit

ROOT=Path(__file__).resolve().parents[2]

def hps(family, gpu):
    from autogluon.common import space
    if family=="tree":
        return {"GBM":{"num_boost_round": space.Int(500,1600),"learning_rate":space.Real(.015,.05),"num_leaves":space.Int(24,64),"min_data_in_leaf":space.Int(8,20),"feature_fraction":space.Real(.7,1.0),"lambda_l1":space.Real(0,.2),"lambda_l2":space.Real(0,.2),"ag_args_fit":{"num_gpus":0}}, "CAT":{"iterations":space.Int(500,1400),"depth":space.Int(6,9),"learning_rate":space.Real(.02,.07),"l2_leaf_reg":space.Real(1,7),"ag_args_fit":{"num_gpus":0}}, "XGB":{"n_estimators":space.Int(500,1400),"max_depth":space.Int(5,9),"learning_rate":space.Real(.02,.07),"subsample":space.Real(.7,1.0),"colsample_bytree":space.Real(.7,1.0),"ag_args_fit":{"num_gpus":int(gpu)}}}
    return {"NN_TORCH":{"num_epochs":space.Int(40,120),"learning_rate":space.Real(3e-4,1.5e-3,log=True),"dropout_prob":space.Real(0,.4),"hidden_size":space.Categorical(64,128,256),"num_layers":space.Int(1,3),"activation":space.Categorical("relu","softrelu"),"ag_args_fit":{"num_gpus":int(gpu)}}}

def run(family, folds, seconds, output, device):
    train=pd.read_csv(ROOT/"artifacts/automl_features_train.csv"); test=pd.read_csv(ROOT/"artifacts/automl_features_test.csv"); output.mkdir(parents=True,exist_ok=True); gpu=device!="cpu"; Predictor=_import_predictor(); probs=[]; summaries=[]
    for fold in folds:
        tr,va=fold_partition(train,fold); d=output/f"fold_{fold}"; model=d/"predictor"; d.mkdir(parents=True,exist_ok=True)
        predictor=Predictor(label=LABEL,problem_type="multiclass",eval_metric="accuracy",path=str(model),verbosity=2,learner_kwargs={"label_count_threshold":1}).fit(train_data=tr.drop(columns=["clip_id","user","fold"]),hyperparameters=hps(family,gpu),hyperparameter_tune_kwargs={"scheduler":"local","searcher":"random","num_trials":999},time_limit=seconds,auto_stack=False,num_bag_folds=0,num_stack_levels=0,fit_weighted_ensemble=False,refit_full=False,num_gpus=int(gpu))
        cols=[c for c in test if c not in {LABEL,"clip_id","user","fold"}]; vp=predictor.predict_proba(va.drop(columns=[LABEL,"clip_id","user","fold"]),as_pandas=True).reindex(columns=range(NUM_CLASSES),fill_value=0); acc=float((vp.to_numpy().argmax(1)==va[LABEL].to_numpy()).mean()); vp.to_csv(d/"validation_probabilities.csv",index=False); lead=predictor.leaderboard(va.drop(columns=["clip_id","user","fold"]),silent=True); lead.to_csv(d/"leaderboard.csv",index=False); tp=predictor.predict_proba(test[cols],as_pandas=True).reindex(columns=range(NUM_CLASSES),fill_value=0).to_numpy(); np.save(d/"test_probabilities.npy",tp); s={"fold":fold,"validation_accuracy":acc,"best_model":predictor.model_best,"seconds_per_fold":seconds}; (d/"summary.json").write_text(json.dumps(s,indent=2)); print(json.dumps(s),flush=True); probs.append(tp); summaries.append(s)
    pred=np.mean(probs,0).argmax(1); test_rows=pd.read_csv(ROOT/"manifests/cv5/test.csv"); submit(test_rows, np.eye(NUM_CLASSES, dtype=np.float32)[pred], ROOT/"../Small-Model-Track/Testing/test_file/test.csv", output/"submission.csv"); (output/"summary.json").write_text(json.dumps({"family":family,"folds":summaries},indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument("--family",choices=("tree","nn"),required=True);p.add_argument("--output",required=True);p.add_argument("--folds",nargs="+",type=int,default=[2,4]);p.add_argument("--seconds-per-fold",type=int,default=7200);p.add_argument("--device",choices=("cuda","cpu"),default="cuda");a=p.parse_args();run(a.family,a.folds,a.seconds_per_fold,Path(a.output),a.device)
if __name__=="__main__":main()
