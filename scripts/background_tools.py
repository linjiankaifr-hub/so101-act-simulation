"""Table-only RGB tint using geometry masks; privileged masks never enter inference."""
import json
import h5py
import numpy as np
import torch
import mujoco
from act_tools import ChunkDataset
from so101_env import ROOT

TRAIN_PALETTE=((1.,1.,1.),(.45,.65,.95),(.95,.65,.45),(.55,.90,.65),
               (.85,.50,.80),(.65,.65,.65),(.35,.80,.80),(.80,.80,.40))
TEST_PALETTE=((.25,.35,.40),(.40,.25,.65),(.35,.60,.30),(.70,.45,.20))

def table_masks(env):
    if env.renderer is None:
        env.renderer=mujoco.Renderer(env.model,env.image_size,env.image_size)
    renderer=env.renderer
    table_id=env.model.body("work_table").id
    geom_ids=np.flatnonzero(env.model.geom_bodyid==table_id)
    result={}
    renderer.enable_segmentation_rendering()
    try:
        for key,camera in (("front","global_d435_camera"),("wrist","wrist_cam")):
            renderer.update_scene(env.data,camera=camera)
            seg=renderer.render().copy()
            result[key]=np.isin(seg[:,:,0],geom_ids)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
    finally:
        renderer.disable_segmentation_rendering()
    return result

def attach_masks(episodes,names):
    index=0
    for name in names:
        manifest=json.loads((ROOT/"data"/name/"manifest.json").read_text())
        with h5py.File(ROOT/"data/background_masks"/(name+".h5")) as f:
            for seed in manifest["train"]:
                for camera in ("front","wrist"):
                    episodes[index][camera+"_table_mask"]=f[str(seed)][camera][:].astype(bool)
                index+=1
    if index!=len(episodes):raise ValueError("mask/episode count mismatch")
    return episodes

class TableChunkDataset(ChunkDataset):
    def __init__(self,episodes,stats,palette=TRAIN_PALETTE,augmentation_seed=42,**kwargs):
        palette=np.asarray(palette,dtype=np.float32)
        if palette.ndim!=2 or palette.shape[1]!=3 or len(palette)==0 or not np.isfinite(palette).all() or (palette<=0).any() or (palette>1).any():
            raise ValueError("palette must contain finite RGB factors in (0,1]")
        for ep in episodes:
            for camera in ("front","wrist"):
                mask=ep.get(camera+"_table_mask")
                if mask is None or mask.shape!=ep[camera].shape[:-1]:
                    raise ValueError("mask must match every source RGB frame")
        super().__init__(episodes,stats,augmentation_seed=augmentation_seed,**kwargs)
        self.palette=palette
        self._table_worker=None
        self._table_rng=None

    def __getitem__(self,index):
        batch=super().__getitem__(index)
        ep_index=int(np.searchsorted(self.ends,index,side="right"))
        frame=int(index-(self.ends[ep_index-1] if ep_index else 0))
        ep=self.episodes[ep_index]
        worker=torch.utils.data.get_worker_info()
        worker_id=-1 if worker is None else worker.id
        if self._table_rng is None or self._table_worker!=worker_id:
            self._table_rng=np.random.default_rng(self.augmentation_seed+2027*(worker_id+1))
            self._table_worker=worker_id
        color=torch.from_numpy(self.palette[self._table_rng.integers(len(self.palette))].copy())[:,None,None]
        for camera in ("front","wrist"):
            key="observation.images."+camera
            rgb=batch[key]*self.std+self.mean
            mask=torch.from_numpy(ep[camera+"_table_mask"][frame].copy())[None].bool()
            batch[key]=torch.where(mask,(rgb*color-self.mean)/self.std,batch[key])
        return batch
