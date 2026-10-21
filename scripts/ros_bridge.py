"""ROS2 node for the same MuJoCo task: RGB/state/commands/reset and optional ACT."""
import argparse,json
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,JointState
from std_msgs.msg import Int32,String
from std_srvs.srv import Trigger
from so101_env import SO101Env,JOINTS,IMAGE_KEYS
from ros_contract import ordered_targets

class Bridge(Node):
    def __init__(self,checkpoint=None,max_steps=0):
        super().__init__("so101_act_lab")
        self.env=SO101Env(image_size=96)
        self.seed=30000
        self.obs=self.env.reset(self.seed)
        self.target=self.env.state
        self.runtime=None
        if checkpoint:
            from act_tools import Runtime
            self.runtime=Runtime(checkpoint)
        self.counter=0; self.max_steps=max_steps
        self.state_pub=self.create_publisher(JointState,"/so101/joint_states",10)
        self.action_pub=self.create_publisher(JointState,"/so101/applied_action",10)
        self.epoch_pub=self.create_publisher(Int32,"/so101/epoch",10)
        self.result_pub=self.create_publisher(String,"/so101/result",10)
        self.cameras=[self.create_publisher(Image,"/so101/"+cam+"/image_raw",5) for cam in ("front","wrist")]
        self.create_subscription(JointState,"/so101/joint_targets",self.command,10)
        self.create_service(Trigger,"/so101/reset",self.reset_service)
        self.create_service(Trigger,"/so101/status",self.status_service)
        self.create_timer(.05,self.tick)
        self.get_logger().info("SO101 bridge ready; units=rad; command frame_id=so101/epoch/"+str(self.env.epoch))

    def command(self,msg):
        try:
            if self.runtime:raise ValueError("external commands disabled while ACT owns control")
            if msg.header.frame_id!="so101/epoch/"+str(self.env.epoch):
                raise ValueError("old or missing command epoch")
            self.target=ordered_targets(msg.name,msg.position)
        except ValueError as error:self.get_logger().warning(str(error))

    def reset_service(self,request,response):
        self.obs=self.env.reset(self.seed)
        self.target=self.env.state
        if self.runtime:self.runtime.reset()
        response.success=True;response.message=json.dumps({"epoch":self.env.epoch})
        return response

    def status_service(self,request,response):
        response.success=True
        response.message=json.dumps({"epoch":self.env.epoch,"steps":self.env.steps,"success":self.env.success,
            "mode":"ACT" if self.runtime else "external_or_hold","seed":self.seed})
        return response

    def tick(self):
        if self.runtime:self.target=self.runtime.action(self.obs)
        msg=JointState()
        msg.header.stamp=self.get_clock().now().to_msg()
        msg.header.frame_id="so101/epoch/"+str(self.env.epoch)
        msg.name=list(JOINTS);msg.position=self.target.tolist()
        self.action_pub.publish(msg)
        self.obs,success,done,_=self.env.step(self.target,render_obs=(self.env.steps+1)%5==0)
        state=JointState()
        state.header=msg.header;state.name=list(JOINTS);state.position=self.env.state.tolist()
        self.state_pub.publish(state)
        self.epoch_pub.publish(Int32(data=self.env.epoch))
        if IMAGE_KEYS[0] in self.obs:
            for pub,key,cam in zip(self.cameras,IMAGE_KEYS,("front","wrist")):
                a=self.obs[key]
                image=Image();image.header.stamp=self.get_clock().now().to_msg();image.header.frame_id=cam+"_optical"
                image.height=a.shape[0];image.width=a.shape[1];image.encoding="rgb8";image.step=a.shape[1]*3
                image.data=a.tobytes();pub.publish(image)
        if success or done:
            self.result_pub.publish(String(data=json.dumps({"seed":self.seed,"success":success,"steps":self.env.steps})))
            self.seed+=1
            self.obs=self.env.reset(self.seed);self.target=self.env.state
            if self.runtime:self.runtime.reset()
        self.counter+=1
        if self.max_steps and self.counter>=self.max_steps:rclpy.shutdown()

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint");p.add_argument("--max-steps",type=int,default=0)
    args=p.parse_args();rclpy.init();node=Bridge(args.checkpoint,args.max_steps)
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        node.env.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=="__main__":main()
