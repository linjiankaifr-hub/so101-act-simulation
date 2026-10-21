"""Seed partitions and development-only failure mining."""
def partitions():
    return {"training":list(range(10000,11000)),"random_extra":list(range(11000,11020)),
            "development":list(range(20000,20200)),"test":list(range(30000,30100)),
            "perturbation":list(range(40000,40030)),
            "brightness_development":list(range(21000,21020)),
            "brightness_test":list(range(50000,50100)),
            "background_development":list(range(22000,22020)),
            "background_test":list(range(60000,60100))}

def failed_development_seeds(rows,count):
    allowed=set(partitions()["development"])
    if any(r["seed"] not in allowed for r in rows):
        raise ValueError("failure mining can only use the development partition")
    chosen=list(dict.fromkeys(r["seed"] for r in rows if not r["success"]))
    if len(chosen)<count:
        raise ValueError(f"need {count} failed development seeds, found {len(chosen)}")
    return chosen[:count]
