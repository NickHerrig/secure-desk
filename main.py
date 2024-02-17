from functools import partial
import logging
import threading
import time

import cv2
import requests
import toml

import supervision as sv
from inference import InferencePipeline


# Load the global configuration file.
config = toml.load("config.toml")

# Create global Supervision annnotators and trackers.
bounding_box = sv.BoundingBoxAnnotator()
label = sv.LabelAnnotator(text_scale=3, text_thickness=2)
tracker = sv.ByteTrack(match_thresh=2)

# Create global variables for application logic state.
alert_timer_started = False
should_send_alert = True
current_code_index = 0
tracked_codes = set()
desk_unlocked = False
alert_state_message = ""
code_sequence_message = ""
code_sequence = config["app"]["code"]
alert_endpoint = config["app"]["alert_endpoint"]



def check_code_sequence(class_ids):
    """
    Checks if the provided class IDs match the expected code sequence. Updates the current code index, 
    the desk unlocked status, the alert status, and logs messages based on the check results.

    Args:
        class_ids (list): List of detected class IDs.

    Global variables:
        current_code_index (int): The current index in the code sequence.
        desk_unlocked (bool): Flag indicating whether the desk is unlocked.
        should_send_alert (bool): Flag indicating whether an alert should be sent.
        code_sequence_message (str): Message indicating the progress of code sequence entry.
    """
    global current_code_index, desk_unlocked, should_send_alert, code_sequence_message
    if code_sequence[current_code_index] in class_ids:
        current_code_index += 1
        msg = f"Correct code entered. Step {current_code_index} of {len(code_sequence)} completed."
        logging.info(msg)
        code_sequence_message = msg
        
        if current_code_index == len(code_sequence):
            desk_unlocked = True
            msg = "Correct sequence entered! Desk is unlocked!"
            logging.info(msg)
            code_sequence_message = msg
            should_send_alert = False
            current_code_index = 0
    else:
        if current_code_index > 0:
            msg = "Incorrect code entered. Resetting sequence."
            logging.info(msg)
            code_sequence_message = msg
            current_code_index = 0


def alert_timer():
    """
    Runs a countdown for 30 seconds. If the desk is unlocked during the countdown, it breaks early.
    If the alert flag is set at the end of the countdown, it sends an alert and updates the alert state message.

    Global variables:
        alert_state_message (str): Message indicating the state of the alert.
        desk_unlocked (bool): Flag indicating whether the desk is unlocked.
        should_send_alert (bool): Flag indicating whether an alert should be sent.
    """
    global alert_state_message, desk_unlocked, should_send_alert
    for i in range(30, 0, -1):
        alert_state_message = f"{i} seconds left."
        time.sleep(1)
        if desk_unlocked:
            break
    
    if should_send_alert:
        send_alert()
        alert_state_message =  "Alert Sent!"

    else:
        alert_state_message = "Desk Unlocked!"


def send_alert():
    """
    Sends an alert message to a specified endpoint using a POST request. The alert message indicates 
    unauthorized access to the desk. 

    Global variable:
        alert_endpoint (str): The endpoint ntfy.sh endpoint to send the alert message

    documentation: https://docs.ntfy.sh/publish/
    """
    global alert_endpoint
    requests.post(alert_endpoint,
        data="Someone is at your desk",
        headers={
            "Title": "Unauthorized access detected",
            "Priority": "urgent",
            "Tags": "warning,skull"
        })
 


def on_prediction(inference_results, frame):

    # Processes inference results into detections, filters by confidence, 
    # extracts class names, and updates the tracker with detections.
    detections = sv.Detections.from_inference(inference_results)
    detections = detections[detections.confidence >0.75]
    class_names = getattr(detections, 'data', {}).get('class_name', []) if hasattr(detections, 'data') else []
    labels = [f"{class_name} - {confidence:.2f}" 
              for class_name, confidence
              in zip(class_names, detections.confidence)]
    tracked_detections = tracker.update_with_detections(detections)    

    # Annotates the frame with labels, bounding boxes, and various text messages.
    annotated_frame = label.annotate(scene=frame.image.copy(), detections=detections, labels=labels)
    annotated_frame = bounding_box.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = sv.draw_text(
        scene=annotated_frame,
        text="Secure Desk v0.01",
        text_anchor= sv.Point(x=1700, y=100),
        text_scale=3,
        background_color=sv.Color(r=255, g=255, b=255),
        text_color=sv.Color(r=0, g=0, b=0),
    )
    annotated_frame = sv.draw_text(
        scene=annotated_frame,
        text=alert_state_message,
        text_anchor= sv.Point(x=1700, y=500),
        text_scale=2,
        background_color=sv.Color(r=75, g=0, b=130),
        text_color=sv.Color(r=230, g=190, b=255),
    )
    annotated_frame = sv.draw_text(
        scene=annotated_frame,
        text=code_sequence_message,
        text_anchor= sv.Point(x=1100, y=1200),
        text_scale=2,
        background_color=sv.Color(r=112, g=128, b=144),
        text_color=sv.Color(r=204, g=255, b=0),
    )

    # Starts an alert timer when a person is detected 
    #and checks the code sequence for each new tracked detection.
    global alert_timer_started, current_entry, code
    if "person" in class_names:
        if not alert_timer_started:
            alert_timer_started = True
            alert_timer_thread = threading.Thread(target=alert_timer,)
            alert_timer_thread.start()

        for tracker_id in tracked_detections.tracker_id:
            if tracker_id not in tracked_codes:
                tracked_codes.add(tracker_id)
                check_code_sequence(tracked_detections.class_id)
   

    # Display the frames
    cv2.imshow("Inference", annotated_frame)
    cv2.waitKey(1)


if __name__ == "__main__":

    # Setup basic logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse rtsp camera conf file values and set the rtsp url
    rtsp_user = config["rtsp"]["user"]
    rtsp_pass = config["rtsp"]["password"]
    camera_ip = config["rtsp"]["camera_ip"]
    rtsp_port = config["rtsp"]["port"]
    rtsp_url = f"rtsp://{rtsp_user}:{rtsp_pass}@{camera_ip}:{rtsp_port}/h264Preview_01_main"

    # Load additional configuration for Inference pipeline
    active_learning = config["app"]["active_learning"]
    roboflow_api_key = config["app"]["roboflow_api_key"]
    roboflow_model = config["app"]["roboflow_model"]
    roboflow_model = config["app"]["roboflow_model"]

    # Initialize the Inference pipeline
    pipeline = InferencePipeline.init(
        model_id=roboflow_model,
        api_key=roboflow_api_key,
        video_reference=rtsp_url, 
        on_prediction=on_prediction,
        active_learning_enabled=active_learning,
    )

    # Start the pipeline
    pipeline.start()
    pipeline.join()