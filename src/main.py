from core.object_tracking import ObjectTracking
from core.object_detection import ObjectDetectionYoloV4
from dotenv import load_dotenv
import os

load_dotenv()

# Get the base file path from the environment variables
BASE_FILE_PATH = os.getenv("BASE_FILE_PATH")

# ---------------- CHANGE THIS IF YOU WANT ----------------
VIDEO_OF_INTEREST = "drone.mp4"  # Select different video to track on.
# ---------------- CHANGE THIS IF YOU WANT ----------------

# Check if BASE_FILE_PATH is defined
if not BASE_FILE_PATH:
    raise ValueError(
        "The environment variable BASE_FILE_PATH is not defined. Please set it to the base file path."
    )

# Check if BASE_FILE_PATH is a valid path
if not os.path.exists(BASE_FILE_PATH):
    raise ValueError(
        "The specified BASE_FILE_PATH does not exist or is not a valid path."
    )

PATH_TO_VIDEO = os.path.join("assets", VIDEO_OF_INTEREST)
PATH_TO_DNN_MODEL = os.path.join(BASE_FILE_PATH, "src", "dnn_model")


def main() -> int:
    detector = ObjectDetectionYoloV4(
        dnn_model_path=PATH_TO_DNN_MODEL,
        nms_threshold=0.4,
        conf_threshold=0.5,
        image_size=608,
    )
    tracker = ObjectTracking(
        detector=detector, path_to_video=PATH_TO_VIDEO, detect_time=3, kf_time=0.5
    )
    tracker.track()
    return 0


if __name__ == "__main__":
    main()
