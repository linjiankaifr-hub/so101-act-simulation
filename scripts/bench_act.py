import json,os,time
from pathlib import Path
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
for k in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","all_proxy"):
    os.environ.pop(k,None)
torch.set_num_threads(2)
features={k:PolicyFeature(type=FeatureType.VISUAL,shape=(3,96,96)) for k in ("observation.images.front","observation.images.wrist")}
features["observation.state"]=PolicyFeature(type=FeatureType.STATE,shape=(6,))
cfg=ACTConfig(input_features=features,output_features={"action":PolicyFeature(type=FeatureType.ACTION,shape=(6,))},
    device="cuda",chunk_size=30,n_action_steps=5,dim_model=256,n_heads=8,dim_feedforward=1024,
    n_encoder_layers=2,n_vae_encoder_layers=2,optimizer_lr=1e-4,optimizer_lr_backbone=1e-5)
policy=ACTPolicy(cfg).cuda()
batch={k:torch.rand(4,*v.shape,device="cuda") for k,v in features.items()}
batch["action"]=torch.rand(4,30,6,device="cuda")
batch["action_is_pad"]=torch.zeros(4,30,dtype=torch.bool,device="cuda")
optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4)
start=time.time()
for _ in range(10):
    loss,info=policy(batch);loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
result={"device":torch.cuda.get_device_name(),"torch":torch.__version__,"batch_size":4,
        "seconds_per_update":(time.time()-start)/10,"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
        "parameters":sum(p.numel() for p in policy.parameters()),"loss":loss.item()}
out=Path(__file__).resolve().parents[1]/"results/act_benchmark.json"
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
