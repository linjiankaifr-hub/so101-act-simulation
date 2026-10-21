"""Official LeRobot ACT with explicit, saved preprocessing and raw-data mirrors."""
import json,os,shutil,hashlib
from pathlib import Path
import h5py,numpy as np,torch
from torch.utils.data import Dataset
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from so101_env import ROOT,IMAGE_KEYS

def action_chunk(actions,index,length):
    indexes=np.arange(index,index+length)
    return actions[np.minimum(indexes,len(actions)-1)].copy(),indexes>=len(actions)

def parameter_fingerprint(parameters):
    h=hashlib.sha256()
    for name,value in sorted(parameters.items()):
        tensor=value.detach().cpu().contiguous()
        h.update(name.encode()+b"\0")
        h.update(str((tuple(tensor.shape),tensor.dtype)).encode()+b"\0")
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()

def vector_stats(x):
    return x.mean(axis=0).astype(np.float32),np.maximum(x.std(axis=0),.025).astype(np.float32)

def make_config(image_size=96):
    inputs={k:PolicyFeature(type=FeatureType.VISUAL,shape=(3,image_size,image_size)) for k in IMAGE_KEYS}
    inputs["observation.state"]=PolicyFeature(type=FeatureType.STATE,shape=(6,))
    return ACTConfig(input_features=inputs,output_features={"action":PolicyFeature(type=FeatureType.ACTION,shape=(6,))},
        device="cuda",chunk_size=30,n_action_steps=30,dim_model=256,n_heads=8,dim_feedforward=1024,
        n_encoder_layers=2,n_vae_encoder_layers=2,optimizer_lr=1e-4,optimizer_lr_backbone=1e-5,
        push_to_hub=False)

def load_episodes(names,split="train"):
    episodes=[]
    for name in names:
        base=ROOT/"data"/name
        manifest=json.loads((base/"manifest.json").read_text())
        ids=manifest[split]
        source=base/"episodes.h5"
        cache=Path.home()/".cache/codex-so101-data"/name/"episodes.h5"
        cache.parent.mkdir(parents=True,exist_ok=True)
        if not cache.exists() or cache.stat().st_size!=source.stat().st_size or cache.stat().st_mtime_ns!=source.stat().st_mtime_ns:
            temporary=cache.with_suffix("."+str(os.getpid())+".tmp")
            shutil.copy2(source,temporary)
            os.replace(temporary,cache)
        print("loading",name,split,"episodes",len(ids),"from ext4 cache",flush=True)
        with h5py.File(cache) as f:
            for seed in ids:
                group=f[str(seed)]
                episodes.append({k:group[k][:] for k in ("state","action","front","wrist")})
    return episodes

class ChunkDataset(Dataset):
    def __init__(self,episodes,stats,chunk=30,brightness_range=None,augmentation_seed=42):
        if brightness_range is not None:
            bounds=np.asarray(brightness_range,dtype=float)
            if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]<=0 or bounds[0]>bounds[1]:
                raise ValueError("brightness range must be two finite positive ordered factors")
            brightness_range=tuple(bounds)
        self.brightness_range=brightness_range
        self.augmentation_seed=int(augmentation_seed)
        self._brightness_worker=None
        self._brightness_rng=None
        self.episodes=episodes
        self.stats=stats
        self.chunk=chunk
        self.ends=np.cumsum([len(e["state"]) for e in episodes])
        self.mean=torch.tensor([.485,.456,.406])[:,None,None]
        self.std=torch.tensor([.229,.224,.225])[:,None,None]

    def __len__(self): return int(self.ends[-1])

    def __getitem__(self,index):
        ep_index=int(np.searchsorted(self.ends,index,side="right"))
        frame=int(index-(self.ends[ep_index-1] if ep_index else 0))
        ep=self.episodes[ep_index]
        actions,pad=action_chunk(ep["action"],frame,self.chunk)
        s=self.stats
        batch={"observation.state":torch.from_numpy((ep["state"][frame]-s["state_mean"])/s["state_std"]),
            "action":torch.from_numpy((actions-s["action_mean"])/s["action_std"]),
            "action_is_pad":torch.from_numpy(pad)}
        factor=1.
        if self.brightness_range is not None:
            worker=torch.utils.data.get_worker_info()
            worker_id=-1 if worker is None else worker.id
            if self._brightness_rng is None or self._brightness_worker!=worker_id:
                self._brightness_rng=np.random.default_rng(self.augmentation_seed+1009*(worker_id+1))
                self._brightness_worker=worker_id
            factor=float(self._brightness_rng.uniform(*self.brightness_range))
        for k,name in zip(IMAGE_KEYS,("front","wrist")):
            image=torch.from_numpy(ep[name][frame].copy()).permute(2,0,1).float()/255.
            if self.brightness_range is not None:image=(image*factor).clamp(0.,1.)
            batch[k]=(image-self.mean)/self.std
        return batch

def stats_from_episodes(episodes):
    result={}
    for key in ("state","action"):
        mean,std=vector_stats(np.concatenate([e[key] for e in episodes]))
        result[key+"_mean"]=mean
        result[key+"_std"]=std
    return result

def save_stats(stats,path):
    Path(path).write_text(json.dumps({k:np.asarray(v).tolist() for k,v in stats.items()},indent=2))

class Runtime:
    def __init__(self,path,device="cuda",execution_steps=None):
        path=Path(path)
        self.policy=ACTPolicy.from_pretrained(path,local_files_only=True).to(device).eval()
        if execution_steps is not None:
            if not 1<=execution_steps<=self.policy.config.chunk_size: raise ValueError("execution steps must fit the predicted chunk")
            self.policy.config.n_action_steps=execution_steps
        self.stats={k:np.array(v,dtype=np.float32) for k,v in json.loads((path/"normalization.json").read_text()).items()}
        self.device=device
        self.mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        self.reset()

    def reset(self):
        self.policy.reset()
        self.calls=0

    def action(self,obs):
        if len(self.policy._action_queue)==0:
            s=self.stats
            batch={"observation.state":torch.from_numpy((obs["observation.state"]-s["state_mean"])/s["state_std"]).to(self.device)[None]}
            for k in IMAGE_KEYS:
                image=torch.from_numpy(obs[k].copy()).permute(2,0,1).float().to(self.device)[None]/255.
                batch[k]=(image-self.mean)/self.std
            self.calls+=1
        else:
            batch={}
        with torch.inference_mode():
            normalized=self.policy.select_action(batch)[0].cpu().numpy()
        return normalized*self.stats["action_std"]+self.stats["action_mean"]
