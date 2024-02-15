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
label = sv.LabelAnnotator()
tracker = sv.ByteTrack()

# global application logic state
alert_timer_started = False
should_send_alert = True
code_sequence = [2, 0]
current_code_index = 0
tracked_codes = set()


def check_code_sequence(tracked_detections):
    """
    Check if the detected prediction matches the next expected code in the sequence.
    If the sequence is fully matched, disable the alert.
    """
    global current_code_index, should_send_alert

    class_ids = tracked_detections.class_id

    if code_sequence[current_code_index] in class_ids:
        current_code_index += 1
        logging.info(f"Correct code entered. Step {current_code_index} of {len(code_sequence)} completed.")
        
        if current_code_index == len(code_sequence):
            logging.info("Correct sequence entered. Desk is unlocked!")
            should_send_alert = False
            # Reset for next time
            current_code_index = 0
    else:
        # If the code is incorrect, reset the sequence
        if current_code_index > 0:
            logging.info(f"Incorrect code entered. Resetting sequence.")
            current_code_index = 0


def alert_timer(alert_endpoint):

    logging.info("You have 10 seconds to crack the code...")

    time.sleep(15)

    global should_send_alert
    if should_send_alert:
        logging.info("Intruder detected, sending alert")
        send_alert(alert_endpoint)
    else:
        logging.info("Desk is unlocked! ")


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

    labels = [f"{class_name} - {confidence}" 
              for class_name, confidence
              in zip(class_names, detections.confidence)]
    

    tracked_detections = tracker.update_with_detections(detections)    

    annotated_frame = label.annotate(scene=frame.image.copy(), detections=detections, labels=labels)
    annotated_frame = bounding_box.annotate(scene=annotated_frame, detections=detections)


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

    # Load configuration file and parse values
    config = toml.load('config.toml')
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
    )

    pipeline.start()
    pipeline.join()
    

if __name__ == "__main__":
    main()
