import imageio.v2 as imageio
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for name,path in [("expert_preview","results/expert_demo.mp4"),("pilot_preview","results/pilot/development-standard/failure.mp4")]:
    r=imageio.get_reader(root/path)
    imageio.imwrite(root/"results"/(name+".png"),r.get_data(20))
    r.close()
