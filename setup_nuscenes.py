"""
nuScenes Dataset Setup Helper
==============================
Run this script to get instructions for downloading the nuScenes mini dataset
and to verify your installation once done.

Usage:
  python setup_nuscenes.py            # print download instructions
  python setup_nuscenes.py --verify   # verify existing install
"""
import sys, os, argparse

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          nuScenes Dataset Download Instructions                  ║
╚══════════════════════════════════════════════════════════════════╝

Step 1 – Register & download
  1. Go to: https://www.nuscenes.org/download
  2. Create a free account and accept the license.
  3. Download "Full dataset (v1.0) – Mini" (~4 GB).
     Files: v1.0-mini.tgz + nuscenes_map_expansion_v1.3.zip (optional)

Step 2 – Extract
  mkdir -p /data/nuscenes
  tar -xzf v1.0-mini.tgz -C /data/nuscenes/
  # Result: /data/nuscenes/v1.0-mini/ with samples/, maps/, etc.

Step 3 – Install SDK
  pip install nuscenes-devkit pyquaternion

Step 4 – Set environment variable
  export NUSCENES_DATAROOT=/data/nuscenes
  # Add to ~/.bashrc to make permanent

Step 5 – Run the pipeline
  python main.py

Note: the pipeline automatically falls back to a synthetic scene
if nuScenes is not found, so you can test without the dataset.
"""


def verify(dataroot):
    print(f"\nVerifying nuScenes at: {dataroot}")
    if not os.path.isdir(dataroot):
        print(f"  ERROR: Directory not found: {dataroot}")
        return False

    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
        print(f"  OK – loaded {len(nusc.sample)} samples, "
              f"{len(nusc.scene)} scenes.")
        sample = nusc.sample[0]
        cam_token = sample["data"]["CAM_FRONT"]
        cam_data  = nusc.get("sample_data", cam_token)
        cam_path  = os.path.join(dataroot, cam_data["filename"])
        if os.path.isfile(cam_path):
            print(f"  OK – camera image found: {cam_path}")
        else:
            print(f"  WARNING: camera image not found: {cam_path}")

        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data  = nusc.get("sample_data", lidar_token)
        lidar_path  = os.path.join(dataroot, lidar_data["filename"])
        if os.path.isfile(lidar_path):
            print(f"  OK – LiDAR file found: {lidar_path}")
        else:
            print(f"  WARNING: LiDAR file not found: {lidar_path}")

        print("\n  nuScenes installation looks good!")
        print("  Run: NUSCENES_DATAROOT=" + dataroot + " python main.py")
        return True

    except ImportError:
        print("  ERROR: nuscenes-devkit not installed.")
        print("         Run: pip install nuscenes-devkit pyquaternion")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing nuScenes installation")
    parser.add_argument("--dataroot", default=os.environ.get("NUSCENES_DATAROOT",
                                                               "/data/nuscenes"))
    args = parser.parse_args()

    if args.verify:
        ok = verify(args.dataroot)
        sys.exit(0 if ok else 1)
    else:
        print(BANNER)
