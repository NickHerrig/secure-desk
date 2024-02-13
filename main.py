import requests

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
        "https://ntfy.sh/roboflow-secure-desk",
        data="Intruder! Intruder!".encode(encoding='utf-8')
    ) 


def main():


if __name__ == "__main__":
    main()