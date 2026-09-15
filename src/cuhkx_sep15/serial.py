"""Four-hour-per-family, failure-isolated Sep15 architecture search."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from cuhkx_sep12.serial import ROOT, Runner, exclusive_lock, now, watch

JOBS=(("ag_tree_search","cuhkx_sep15.automl_search",lambda out:["--family","tree","--output",out,"--seconds-per-fold","7200"]),("ag_nn_search","cuhkx_sep15.automl_search",lambda out:["--family","nn","--output",out,"--seconds-per-fold","7200"]),("sensor_gru_search","cuhkx_sep15.sequence_search",lambda out:["--kind","gru","--output",out,"--seconds-per-fold","7200"]),("sensor_transformer_search","cuhkx_sep15.sequence_search",lambda out:["--kind","transformer","--output",out,"--seconds-per-fold","7200"]))
def run(directory,timeout_hours=4):
 r=Runner(directory,timeout_hours); results={}
 for name,module,args in JOBS:
  result=r.job(name,module,args,required=("summary.json","submission.csv")); record={"status":"success" if result else "failed","submission":str(result/"submission.csv") if result else None}
  if result:
   target=r.directory/"submissions"/f"{name}.csv";target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(result/"submission.csv",target);record["submission"]=str(target)
  r.state["methods"][name]=record;r.flush();results[name]=record
 r.state.update(status="finished",current_job=None,finished=now());r.flush();print(json.dumps(results,indent=2));return 0 if all(x["status"]=="success" for x in results.values()) else 1
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",default="artifacts/sep15/serial");p.add_argument("--timeout-hours",type=float,default=4.25,help="per-family watchdog; 4h search plus 15m for saving predictions");p.add_argument("--watch",action="store_true");p.add_argument("--interval",type=float,default=5);a=p.parse_args();d=(ROOT/a.run_dir).resolve()
 if a.watch: watch(d,max(1,a.interval));return
 d.mkdir(parents=True,exist_ok=True)
 with exclusive_lock(d/"runner.lock"):raise SystemExit(run(d,a.timeout_hours))
if __name__=="__main__":main()
