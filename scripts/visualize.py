import rerun as rr
from decord import VideoReader, cpu
import numpy as np
import time
import argparse

def main(args):
    # 1. 初始化 Rerun
    rr.init("Visualization", spawn=False)
    server_uri = rr.serve_grpc(grpc_port=9876)
    rr.serve_web_viewer(web_port=9090, connect_to=server_uri)

    # 2. 加载数据
    vr = VideoReader(args.rgb_path, ctx=cpu(0))
    rgb = vr[:].asnumpy()
    xyz = np.load(args.xyz_path)['xyz']  # (T, H, W, 3)
    
    assert len(rgb) == len(xyz)
    num_frames = len(rgb)
    

    for frame_idx in range(num_frames):
        colors = rgb[frame_idx].reshape(-1, 3)
        points = xyz[frame_idx].reshape(-1, 3)

        rr.set_time("frame_idx", sequence=frame_idx)
        
        # log origin and axes
        rr.log(
            "world/origin",
            rr.Points3D(
                np.array([[0.0, 0.0, 0.0]]),
                colors=np.array([[255, 255, 255]], dtype=np.uint8),
            )
        )

        # 记录 2D 图像
        rr.log("world/rgb", rr.Image(rgb[frame_idx]))
        axis_strips = [
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),  # X
            np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),  # Y
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]),  # Z
        ]
        axis_colors = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
        )
        rr.log(
            "world/axes",
            rr.LineStrips3D(axis_strips, colors=axis_colors),
        )

        # 记录彩色 3D 点云
        rr.log(
            "world/xyz",
            rr.Points3D(
                points, 
                colors=colors
            )
        )
        
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", type=str, required=True)
    parser.add_argument("--xyz_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
