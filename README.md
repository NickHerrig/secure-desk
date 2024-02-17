
# Secure Desk

## Description

Secure desk is a real time computer vision application that secures a desk/office behind a passcode and alerting system. The main functionality of the application is to send a push notification if a person is detected and they do not enter a passcode within 30 seconds of detection. Passcodes are configurable in length, and entered via digits on your hand.

## Demo Video

[![Secure Desk Demo](https://img.youtube.com/vi/BvkXE6Y6L-A/0.jpg)](https://www.youtube.com/watch?v=BvkXE6Y6L-A)

## Installation

To set up the Secure Desk, follow these steps:

1. **Clone the repository:**

```bash
git clone https://github.com/NickHerrig/secure-desk.git
cd secure-desk
```

2. **Install dependencies:**

Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

This command installs all necessary libraries, including `Inference`, `Supervision`, `toml`, and others required for the operation of the system.

3. **Configuration:**

Edit the `config.toml` file to match your environment settings, including RTSP camera credentials, alert endpoint, and other relevant configurations.

## Usage

To run the Secure Desk System, execute the following command:

```bash
python main.py
```

Ensure your RTSP stream is operational and the `config.toml` file is correctly set up before starting the system.

## Contributing

Contributions to the Secure Desk System are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.