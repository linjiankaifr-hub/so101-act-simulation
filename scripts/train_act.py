"""Train the official LeRobot ACTPolicy; no scripted actions in policy inference."""
import argparse,json,os,time,random
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from act_tools import ACTPolicy,make_config,load_episodes,ChunkDataset,stats_from_episodes,save_stats,parameter_fingerprint
from so101_env import ROOT
from stage_artifacts import recover_incomplete

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--name",default="base")
    p.add_argument("--datasets",nargs="+",default=["base"])
    p.add_argument("--steps",type=int,default=6000)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--brightness-range",nargs=2,type=float,default=None)
    p.add_argument("--table-augmentation",action="store_true")
    a=p.parse_args()
    for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"): os.environ.pop(k,None)
    torch.set_num_threads(2)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cudnn.benchmark=True
    destination=ROOT/"checkpoints"/a.name
    recover_incomplete(destination,"training_summary.json",ROOT/"checkpoints")
    destination.mkdir(parents=True,exist_ok=True)
    log=ROOT/"logs"/("train-"+a.name+".jsonl")
    if log.exists():
        import uuid
        log.rename(log.with_name(log.name+".interrupted-"+uuid.uuid4().hex[:8]))
    episodes=load_episodes(a.datasets)
    validation=load_episodes(["base"],"validation")
    stats=stats_from_episodes(episodes)
    if a.table_augmentation:
        if a.brightness_range is not None:raise ValueError("This experiment isolates table color; brightness must be disabled")
        from background_tools import TableChunkDataset,attach_masks
        train_ds=TableChunkDataset(attach_masks(episodes,a.datasets),stats,augmentation_seed=a.seed)
    else:
        train_ds=ChunkDataset(episodes,stats,brightness_range=a.brightness_range,augmentation_seed=a.seed)
    val_ds=ChunkDataset(validation,stats)
    loader=DataLoader(train_ds,batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True)
    val_subset=Subset(val_ds,np.linspace(0,len(val_ds)-1,200,dtype=int).tolist())
    val_loader=DataLoader(val_subset,batch_size=a.batch_size,shuffle=False,num_workers=0)
    cfg=make_config()
    policy=ACTPolicy(cfg).cuda()
    initial_parameters={n:p.detach().cpu().clone() for n,p in policy.named_parameters()}
    optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=cfg.optimizer_lr,weight_decay=cfg.optimizer_weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.steps,eta_min=1e-6)
    scaler=torch.amp.GradScaler("cuda")
    start_step=0
    iterator=iter(loader)
    metrics=[];start=time.time();best=float("inf")
    log=ROOT/"logs"/("train-"+a.name+".jsonl")
    config={"initial_parameter_sha256":parameter_fingerprint(initial_parameters),"validation_augmentation":None,"run":vars(a),"policy":cfg.to_dict() if hasattr(cfg,"to_dict") else str(cfg),
        "training_episodes":len(episodes),"validation_episodes":len(validation),"training_frames":len(train_ds),
        "vision_pretraining":"torchvision ResNet18 ImageNet1K_V1; ACT/transformer trained from random initialization",
        "input_keys":list(cfg.input_features),"normalization":"ImageNet mean/std for RGB; training-only mean/std for state/action"}
    (destination/"run.json").write_text(json.dumps(config,indent=2,default=str))
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step,a.steps):
        try: batch=next(iterator)
        except StopIteration:
            iterator=iter(loader);batch=next(iterator)
        batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
        policy.train()
        with torch.autocast("cuda",dtype=torch.float16):
            loss,info=policy(batch)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);scheduler.step()
        if (step+1)%100==0:
            row={"step":step+1,"loss":loss.item(),**info,"elapsed":time.time()-start,
                "updates_per_second":(step+1-start_step)/(time.time()-start)}
            print(json.dumps(row),flush=True)
            with log.open("a") as f: f.write(json.dumps(row)+"\n")
        if (step+1)%1000==0 or step+1==a.steps:
            policy.eval();values=[]
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                for vi,v in enumerate(val_loader):
                    v={k:t.cuda() for k,t in v.items()}
                    pred=policy.predict_action_chunk(v)
                    error=(pred-v["action"]).abs()*~v["action_is_pad"].unsqueeze(-1)
                    values.append(error.mean().item())
                    if vi>=24:break
            score=float(np.mean(values))
            snapshot=destination/("step_"+str(step+1))
            policy.save_pretrained(snapshot)
            save_stats(stats,snapshot/"normalization.json")
            (snapshot/"experiment.json").write_text(json.dumps({"training_step":step+1,"validation_l1":score,"datasets":a.datasets}))
            # Training restarts from the fixed seed after interruption; no inexact partial-resume mode.
            (destination/"latest.txt").write_text(str(snapshot))
            if score<best:
                best=score;(destination/"best.txt").write_text(str(snapshot))
            print(json.dumps({"checkpoint":str(snapshot),"validation_l1":score}),flush=True)
    delta_sq=sum(float((p.detach().cpu()-initial_parameters[n]).square().sum()) for n,p in policy.named_parameters())
    report={"parameter_change_l2":delta_sq**.5,"steps":a.steps,"training_seconds":time.time()-start,"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
        "training_episodes":len(episodes),"initial_parameter_sha256":config["initial_parameter_sha256"],"seed":a.seed,"latest":str(snapshot),"best_validation_l1":best}
    (destination/"training_summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=="__main__": main()
