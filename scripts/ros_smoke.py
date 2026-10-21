"""Two-process ROS smoke test: receive RGB/state and reset the live environment."""
import json,subprocess,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,JointState
from std_srvs.srv import Trigger
root=Path(__file__).resolve().parents[1]
log=(root/"logs/ros-bridge.log").open("w")
process=subprocess.Popen(["bash",str(root/"scripts/ros.sh"),"--max-steps","500"],stdout=log,stderr=subprocess.STDOUT)
rclpy.init();node=Node("so101_smoke_client")
counts={"front":0,"wrist":0,"state":0}
def image_callback(msg,key):
    assert msg.width==96 and msg.height==96 and msg.encoding=="rgb8"
    assert len(msg.data)==96*96*3
    counts[key]+=1
def state_callback(msg):
    assert len(msg.position)==6;counts["state"]+=1
subs=[node.create_subscription(Image,"/so101/"+cam+"/image_raw",lambda msg,k=cam:image_callback(msg,k),10) for cam in ("front","wrist")]
subs.append(node.create_subscription(JointState,"/so101/joint_states",state_callback,10))
client=node.create_client(Trigger,"/so101/reset")
deadline=time.time()+50
try:
    while time.time()<deadline and min(counts.values())<2:
        if process.poll() is not None:raise RuntimeError("bridge exited")
        rclpy.spin_once(node,timeout_sec=.2)
    assert min(counts.values())>=2,counts
    assert client.wait_for_service(timeout_sec=5)
    future=client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node,future,timeout_sec=10)
    response=future.result()
    assert response and response.success
    report={"passed":True,"received":counts,"reset":json.loads(response.message),"domain":47}
    (root/"results/ros_smoke.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
finally:
    process.terminate()
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill();process.wait()
    log.close();node.destroy_node();rclpy.shutdown()
