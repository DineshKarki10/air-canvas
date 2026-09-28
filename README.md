# Air Canvas — Gesture-Controlled Drawing

An interactive computer vision application built with Python that allows users to draw in the air using hand gestures tracked via webcam in real-time. 

---

## Features

* **Real-Time Hand Tracking:** Powered by Google's MediaPipe framework to precisely track hand landmarks and finger coordinates.
* **Gesture Recognition:** Switch between drawing modes and selection modes using simple finger arrangements (index vs. middle finger).
* **Color Palette & Tools:** Select different colors (Blue, Green, Magenta) or use an Eraser to clean up mistakes.
* **Screen Clear:** Instantly wipe the canvas clean using a dedicated top-bar gesture zone.

---

## Tech Stack

* **Python** (Core programming language)
* **OpenCV** (Webcam management, image processing, and rendering)
* **MediaPipe** (Hand landmark detection model)
* **NumPy** (Array manipulation for the drawing canvas)

---

## Getting Started Locally

Follow these steps to get the project running on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/air-canvas.git](https://github.com/your-username/air-canvas.git)
cd air-canvas