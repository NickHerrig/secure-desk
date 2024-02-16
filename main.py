import cv2
import requests
from functools import partial
import time
import threading
import logging

import supervision as sv
import toml
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

# global supervision annnotators
bounding_box = sv.BoundingBoxAnnotator()
label = sv.LabelAnnotator(text_scale=3, text_thickness=2)
tracker = sv.ByteTrack()

# global application logic state
alert_timer_started = False
should_send_alert = True
code_sequence = [2, 0, 5]
current_code_index = 0
tracked_codes = set()
desk_unlocked = False
alert_state_message = ""
code_sequence_message = ""


def check_code_sequence(tracked_detections):
    """
    Check if the detected prediction matches the next expected code in the sequence.
    If the sequence is fully matched, disable the alert.
    """
    global current_code_index, should_send_alert, code_sequence_message

    class_ids = tracked_detections.class_id

    if code_sequence[current_code_index] in class_ids:
        current_code_index += 1
        msg = f"Correct code entered. Step {current_code_index} of {len(code_sequence)} completed."
        logging.info(msg)
        code_sequence_message = msg
        if current_code_index == len(code_sequence):

            global desk_unlocked
            desk_unlocked = True
            msg = "Correct sequence entered! Desk is unlocked!"
            logging.info(msg)
            code_sequence_message = msg
            should_send_alert = False
            current_code_index = 0
    else:
        # If the code is incorrect, reset the sequence
        if current_code_index > 0:
            msg = "Incorrect code entered. Resetting sequence."
            logging.info(msg)
            code_sequence_message = msg
            current_code_index = 0


def alert_timer(alert_endpoint):

    for i in range(30, 0, -1):
        global alert_state_message
        alert_state_message = f"{i} seconds left."
        time.sleep(1)

        global desk_unlocked
        if desk_unlocked:
            break
    
    global should_send_alert
    if should_send_alert:
        send_alert(alert_endpoint)
        alert_state_message =  "Alert Sent!"

    else:
        alert_state_message = "Desk Unlocked!"

def send_alert(alert_endpoint):
    """
    Send an alert message to a specified URL.

    This function posts an alert message to a specified URL using the requests library. 
    The message is encoded in UTF-8 format before being sent.

    Parameters:
    message (str): The alert message to be sent.

    Returns:
    Response: The response from the server after the post request is made.
    """    
    requests.post(alert_endpoint,
        data="Someone is at your desk",
        headers={
            "Title": "Unauthorized access detected",
            "Priority": "urgent",
            "Tags": "warning,skull"
        })
 


def on_prediction(alert_endpoint, inference_results, frame):

    detections = sv.Detections.from_inference(inference_results)

    class_names = getattr(detections, 'data', {}).get('class_name', []) if hasattr(detections, 'data') else []

    labels = [f"{class_name} - {confidence:.2f}" 
              for class_name, confidence
              in zip(class_names, detections.confidence)]
    

    tracked_detections = tracker.update_with_detections(detections)    

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


    if "person" in class_names:

        global alert_timer_started
        if not alert_timer_started:

            alert_timer_started = True

            alert_timer_thread = threading.Thread(target=alert_timer, args=("https://ntfy.sh/roboflow-secure-desk",))

            alert_timer_thread.start()

        global current_entry
        global code


        for tracker_id in tracked_detections.tracker_id:
            if tracker_id not in tracked_codes:
                tracked_codes.add(tracker_id)
                check_code_sequence(tracked_detections)
   

    cv2.imshow("Inference", annotated_frame)
    cv2.waitKey(1)


def main():

    # Setup basic logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = toml.load("config.toml")

    # Load configuration file and parse values
    rtsp_user = config["rtsp"]["user"]
    rtsp_pass = config["rtsp"]["password"]
    camera_ip = config["rtsp"]["camera_ip"]
    rtsp_port = config["rtsp"]["port"]

    # Use rtsp values to create the rtsp url for use in inference pipeline
    rtsp_url = f"rtsp://{rtsp_user}:{rtsp_pass}@{camera_ip}:{rtsp_port}/h264Preview_01_main"

    # Load additional configuration for Inference pipeline
    active_learning = config["app"]["active_learning"]
    roboflow_api_key = config["app"]["roboflow_api_key"]
    roboflow_model = config["app"]["roboflow_model"]
    roboflow_model = config["app"]["roboflow_model"]
    alert_endpoint = config["app"]["alert_endpoint"]

    # create a function partial to pass confi to callback.
    alert_config_partial = partial(on_prediction, alert_endpoint)

    # Initialize the Inference pipeline
    pipeline = InferencePipeline.init(
        model_id=roboflow_model,
        api_key=roboflow_api_key,
        video_reference=rtsp_url, 
        on_prediction=alert_config_partial,
        active_learning_enabled=active_learning,
        confidence=.80
    )

    pipeline.start()
    pipeline.join()
    

if __name__ == "__main__":
    main()
