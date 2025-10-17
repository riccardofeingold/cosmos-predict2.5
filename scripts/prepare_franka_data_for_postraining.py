import h5py
import json
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial.transform import Rotation as R


def absolute_to_relative_poses(poses):
    """
    Convert a sequence of absolute poses to relative poses between consecutive timesteps.

    Parameters
    ----------
    poses : np.ndarray of shape (N, 7)
        Sequence of poses [x, y, z, qx, qy, qz, qw] at each time step.

    Returns
    -------
    rel_poses : np.ndarray of shape (N-1, 6)
        Relative poses [dx, dy, dz, droll, dpitch, dyaw] between consecutive steps.
    """
    assert poses.shape[1] == 7, "Each pose must have 7 elements (x, y, z, qx, qy, qz, qw)."

    # Extract translations and rotations
    t = poses[:, :3]
    q = poses[:, 3:]

    # Convert to Rotation objects
    R_all = R.from_quat(q)

    # Compute relative rotations (R_rel = R_i^T * R_{i+1})
    R_rel = R_all[:-1].inv() * R_all[1:]
    euler_rel = R_rel.as_euler('xyz', degrees=False)

    # Compute relative translations in local frame
    dt = t[1:] - t[:-1]
    t_rel = np.array([R_all[i].inv().apply(dt[i]) for i in range(len(dt))])

    # Combine translation + euler
    rel_poses = np.hstack((t_rel, euler_rel))
    return rel_poses

def process_qpos_franka(qposes):
    quats = qposes[:, 3:]
    rots = R.from_quat(quats)
    eulers = rots.as_euler('xyz', degrees=False)
    new_qposes = np.hstack((qposes[:, :3], eulers))

    return new_qposes

def process_qpos_hand(x):
    """
    Map values from [-1, 1] range to [0, 1].
    and flips the sign such that 0 means open and 1 means closed.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s) in range [-1, 1].

    Returns
    -------
    y : float or np.ndarray
        Mapped value(s) in range [0, 1].
    """
    return (-x + 1) / 2

def h5_to_bridge_format(h5_path, base_dir="processed_sim_franka_data"):
    """
    Converts an HDF5 file into:
      - MP4 video under datasets/bridge/videos/
      - JSON metadata under datasets/bridge/annotations/
    following the Bridge dataset structure.
    """
    h5_path = Path(h5_path)
    base_dir = Path(base_dir)
    videos_dir = base_dir / "videos"
    ann_dir = base_dir / "annotations"
    videos_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    episode_name = h5_path.stem
    video_path = videos_dir / f"{episode_name}.mp4"
    json_path = ann_dir / f"{episode_name}.json"

    try:
        with h5py.File(h5_path, "r") as f:
            # --- Extract frames and save as MP4 ---
            front_frames = f["observations"]["images"]["oakd_front_view_images"]["color"]
            n_frames_front = front_frames.shape[0]
            fps = 10  # Adjust if needed or read from attributes

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (720, 720))

            for i in range(n_frames_front):
                frame = front_frames[i]
                # resize each frame to 720px720p
                frame = cv2.resize(frame, (720, 720))
                frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            video_writer.release()

            # --- Build JSON metadata ---
            metadata = {}
            mapping = {
                "qpos_franka": "state",
                "qpos_hand": "continuous_gripper_state",
                "action": None,
            }
            for h5_key, json_key in mapping.items():
                if h5_key in f["observations"]:
                    arr = np.array(f["observations"][h5_key])
                    if h5_key == "qpos_franka":
                        arr = process_qpos_franka(arr)
                    elif h5_key == "qpos_hand":
                        arr = process_qpos_hand(arr)
                    metadata[json_key] = arr.tolist()
                elif h5_key == "action":
                    # Compute actions as differences in qpos_franka
                    qpos = np.array(f["observations"]["qpos_franka"])
                    actions = absolute_to_relative_poses(qpos)
                    # Pad the first action with zeros to match lengths
                    actions = np.vstack((np.zeros((1, actions.shape[1])), actions))
                    metadata["action"] = actions.tolist()

            metadata["action"] = np.hstack((np.array(metadata["action"]), np.array(metadata["continuous_gripper_state"]))).tolist()
            metadata["state"] = np.hstack((np.array(metadata["state"]), np.array(metadata["continuous_gripper_state"]))).tolist()

            with open(json_path, "w") as jf:
                json.dump(metadata, jf, indent=2)

        return f"✅ {episode_name}: done ({n_frames_front} frames)"

    except Exception as e:
        return f"❌ {episode_name}: {e}"


def process_all_h5_parallel(input_dir, base_dir="datasets/bridge", max_workers=4):
    """
    Processes all .h5 files in a directory in parallel,
    converting them to the Bridge dataset format.
    """
    input_dir = Path(input_dir)
    h5_files = sorted(input_dir.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {input_dir}")
        return

    print(f"Found {len(h5_files)} HDF5 files — processing with {max_workers} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(h5_to_bridge_format, h5, base_dir): h5 for h5 in h5_files}
        for future in as_completed(futures):
            result = future.result()
            print(result)
            results.append(result)

    print("\nAll files processed.")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_bridge_dataset_parallel.py <input_folder> [base_dir] [max_workers]")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    base_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_sim_franka_data"
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    process_all_h5_parallel(input_dir, base_dir, max_workers)
