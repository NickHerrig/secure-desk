import requests
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import onnxruntime as ort


def send_alert(message):
    """
    Send an alert message to a specified URL.

    This function posts an alert message to a specified URL using the requests library. 
    The message is encoded in UTF-8 format before being sent.

    Parameters:
    message (str): The alert message to be sent.

    Returns:
    Response: The response from the server after the post request is made.
    """
    return requests.post(
        "", # Redacted 
        data=message.encode(encoding='utf-8')
    ) 


def main():

    print("Available Providers: ", ort.get_available_providers())

    rtsp_user = "" # Redacted
    rtsp_password = "" # Redacted
    camera_ip = "" # Redacted


    pipeline = InferencePipeline.init(
        model_id="coco/6",
        video_reference=f"rtsp://{rtsp_user}:{rtsp_password}@{camera_ip}:554/h264Preview_01_main", 
        on_prediction=render_boxes,
        api_key="", # Redacted
        max_fps=10,
        confidence=0.75
    )

    pipeline.start()
    pipeline.join()


    # TODO Hand Signal For Enabling Lock

    # TODO Hand Signal Unlocking the Area

    # There will be some global state if the area is locked/unlocked. 

    # If the desk is Locked, start alert timer, and watch for  

    # We'll want a tracker to track hand signals, and people. 

    # The main app will detect people in the space, and when detected will kick off an alert timer. 
    

if __name__ == "__main__":
    main()