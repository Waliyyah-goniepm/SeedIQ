# SeedIQ
An automated computer vision system designed to grade Cassia Tora seeds in real-time. This project replaces manual inspection with an AI solution that detects Grade A (Premium), Grade B (Defective), and Impurities (Stones/Sticks).

The system is built on a YOLOv8n architecture and deployed via a user-friendly Streamlit interface.

Key Features

- Real-Time Detection: Live inference via webcam or image upload.

- Triple-Cut Grading Logic: sophisticated decision engine that categorizes batches as Premium, Low Quality, or Too Dirty based on configurable thresholds.

- Color Fidelity Engine: Custom OpenCV processing pipeline to ensure accurate color representation (fixing the BGR/RGB conflict).

- Safety Interlocks: "Minimum Seed Count" logic prevents the user from scanning insufficient samples.

- Dynamic Calibration: Sidebar sliders allow non-technical users to adjust strictness (e.g., changing Max Impurity from 15% to 10%) without touching code.

The "Dump & Swipe" Protocol

To overcome the physical limitation of occlusion (seeds hiding defects in deep piles), this project enforces a strict physical protocol:

1. Dump: Pour a small handful of seeds onto a white background.

2. Swipe: Use one hand motion to spread seeds into a Monolayer.

3. Scan: Capture the image.

Note: The model is optimized for spaced/touching seeds, not deep heaps.

Performance Metrics

The model was trained on a custom dataset of ~500 annotated instances using the "Fresh Start" strategy (High-Quality Data Focus).

Class

Precision

Recall

Status

Grade A (Premium)

0.85

1.00

Zero loss of good product

Grade B (Defect)

1.00

0.84

Zero false rejection of good seeds

Impurity

0.95

0.64

High confidence detection

Overall mAP



~0.90

Production Ready

Installation & Usage

1. Clone the Repository

git clone [https://github.com/your-username/cassia-grader.git](https://github.com/your-username/cassia-grader.git)
cd cassia-grader


2. Install Dependencies

It is recommended to use a virtual environment.

pip install -r requirements.txt


(Note: Requires ultralytics, streamlit, opencv-python, numpy<2.0)

3. Run the App

streamlit run app.py


 Project Structure

cassia-grader/
├── app.py              # Main application logic (Streamlit)
├── best.pt             # Trained YOLOv8 model weights
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation


Technical Challenges Solved

The "Blue Seed" Anomaly: Solved an OpenCV BGR vs. Streamlit RGB color space conflict by implementing a permanent numpy-based channel flip [:, :, ::-1] in the rendering loop.

The Pile Problem: Initially, models failed on piled seeds (40% accuracy). We pivoted to a single-stage detector with the "Dump & Swipe" user protocol, boosting accuracy to 90%.

Confidence Hallucination: Tuned the confidence threshold to 0.50 to prevent the model from guessing on textured backgrounds.

 Acknowledgments

YOLOv8 by Ultralytics for the detection architecture.

Streamlit for the rapid application framework.

Roboflow for dataset management and annotation tools.

Author: Waliyyah Abdurrozaq
