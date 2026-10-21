import ast
from pathlib import Path
from so101_env import ROOT

def test_training_exposes_no_inexact_resume_mode():
    source=(ROOT/"scripts/train_act.py").read_text()
    tree=ast.parse(source)
    flags=[arg.value for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Attribute) and node.func.attr=="add_argument"
           for arg in node.args if isinstance(arg,ast.Constant)]
    assert "--resume" not in flags, "unsupported partial resume must not silently change the training schedule"
