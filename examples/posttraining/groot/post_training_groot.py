from datetime import datetime
import subprocess
from loguru import logger


def run_command(cmd: str, step_name: str = "", max_retry_counter: int = 3) -> subprocess.CompletedProcess:
    retry_counter = 0
    while retry_counter < max_retry_counter:
        try:
            logger.info(f"[{step_name}] Running command: {cmd}")
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            output_lines = []
            for line in process.stdout:
                print(line, end="")  # Realtime print to console
                output_lines.append(line)

            process.wait()
            returncode = process.returncode

            if returncode == 0:
                logger.success(f"[{step_name}] Command succeeded.")
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="".join(output_lines), stderr=None)

            retry_counter += 1
            logger.warning(f"[{step_name}] Attempt {retry_counter} failed with return code {returncode}")

        except Exception as e:
            retry_counter += 1
            logger.warning(f"[{step_name}] Exception during attempt {retry_counter}: {e}")

    logger.error(f"[{step_name}] Command failed after {max_retry_counter} attempts.")
    return subprocess.CompletedProcess(args=cmd, returncode=1, stdout=None, stderr="Command failed.")


def step1_prepare_train_data():
    command = """huggingface-cli download nvidia/GR1-100 --repo-type dataset --local-dir datasets/benchmark_train/hf_gr1/ && \
    mkdir -p datasets/benchmark_train/gr1/videos && \
    mv datasets/benchmark_train/hf_gr1/gr1/*mp4 datasets/benchmark_train/gr1/videos && \
    mv datasets/benchmark_train/hf_gr1/metadata.csv datasets/benchmark_train/gr1 && \
    # python -m scripts.get_t5_embeddings_from_groot_dataset --dataset_path datasets/benchmark_train/gr1 --meta_csv datasets/benchmark_train/gr1/metadata.csv --cache_dir checkpoints/google-t5/t5-11b
    """
    result = run_command(command, step_name="step1_prepare_train_data")
    return result


def step2_train_gr1():
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    command = f"""torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_video2world_training_2b_groot_gr1_480 job.name=test_job_{time_str} trainer.max_iter=100
    """
    result = run_command(command, step_name="step2_train_gr1")
    return result


def step3_inference():
    command = """python -m examples.video2world_gr00t --model_size 14B --gr00t_variant gr1 --prompt \"Use the right hand to pick up rubiks cube from from the bottom of the three-tiered wooden shelf to to the top of the three-tiered wooden shelf.\" --input_path \"assets/sample_gr00t_dreams_gr1/8_Use_the_right_hand_to_pick_up_rubik\'s_cube_from_from_the_bottom_of_the_three-tiered_wooden_shelf_to_to_the_top_of_the_three-tiered_wooden_shelf..png\" --prompt_prefix \"\"
    """
    result = run_command(command, step_name="step3_inference (single gpu)")
    return result


def step5_inference_multi_gpu():
    multi_gpu_command = f"""torchrun --nproc_per_node=8 --master_port=12341 \
        -m examples.video2world_gr00t \
        --model_size 14B \
        --gr00t_variant droid \
        --prompt \"A multi-view video shows that a robot pick the lid and put it on the pot The video is split into four views: The top-left view shows the robotic arm from the left side, the top-right view shows it from the right side, the bottom-left view shows a first-person perspective from the robot's end-effector (gripper), and the bottom-right view is a black screen (inactive view). The robot pick the lid and put it on the pot\" \
        --input_path assets/sample_gr00t_dreams_droid/episode_000408.png \
        --prompt_prefix \"\" --num_gpus 8 \
        --save_path output/generated_video_droid.mp4
    """
    result = run_command(multi_gpu_command, step_name="step3_inference (multi gpu)")
    return result


def step4_inference_multi_gpu_gr1():
    multi_gpu_command = """torchrun --nproc_per_node=8 --master_port=12341 \
        -m examples.video2world_gr00t \
        --model_size 14B \
        --gr00t_variant gr1 \
        --prompt \"Use the right hand to pick up rubik's cube from from the bottom of the three-tiered wooden shelf to to the top of the three-tiered wooden shelf.\" \
        --input_path "assets/sample_gr00t_dreams_gr1/8_Use_the_right_hand_to_pick_up_rubik\'s_cube_from_from_the_bottom_of_the_three-tiered_wooden_shelf_to_to_the_top_of_the_three-tiered_wooden_shelf..png" \
        --prompt_prefix \"\" --num_gpus 8 \
        --save_path output/generated_video_gr1.mp4
    """
    result = run_command(multi_gpu_command, step_name="step3_inference (multi gpu)")
    return result


def step6_inference_benchmark():
    command = f"""rm -rf dream_gen_benchmark/ && \
    huggingface-cli download nvidia/EVAL-175 --repo-type dataset --local-dir dream_gen_benchmark && \
    python -m scripts.prepare_batch_input_json \
        --dataset_path dream_gen_benchmark/gr1_object/ \
        --save_path results/dream_gen_benchmark/cosmos_predict2_14b_gr1_object/ \
        --output_path dream_gen_benchmark/gr1_object/batch_input.json && \
    python -m scripts.prepare_batch_input_json \
        --dataset_path dream_gen_benchmark/gr1_behavior/ \
        --save_path results/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/ \
        --output_path dream_gen_benchmark/gr1_behavior/batch_input.json && \
    python -m scripts.prepare_batch_input_json \
        --dataset_path dream_gen_benchmark/gr1_env/ \
        --save_path results/dream_gen_benchmark/cosmos_predict2_14b_gr1_env/ \
        --output_path dream_gen_benchmark/gr1_env/batch_input.json
    """
    result = run_command(command)
    return result


def test_full_pipeline_in_predict2():
    """
    This function tests the full pipeline in predict2.
    """
    result = step1_prepare_train_data()
    if result.returncode != 0:
        raise SystemError("step1_prepare_train_data failed")
    result = step2_train_gr1()
    if result.returncode != 0:
        raise SystemError("step2:training groot failed")
    result = step3_inference()
    if result.returncode != 0:
        raise SystemError("step3:inference failed")
    result = step4_inference_multi_gpu_gr1()
    if result.returncode != 0:
        raise SystemError("step4:inference on multi gpu failed")
    result = step5_inference_multi_gpu()
    if result.returncode != 0:
        raise SystemError("step5:inference on multi gpu failed")
    result = step6_inference_benchmark()
    if result.returncode != 0:
        raise SystemError("step6:inference on benchmark failed")


if __name__ == "__main__":
    test_full_pipeline_in_predict2()
