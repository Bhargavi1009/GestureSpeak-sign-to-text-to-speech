import gradio as gr
import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Load your trained model
model = load_model("cnn8grps_rad1_model.h5")

# Hand detector
hd = HandDetector(maxHands=1)


def predict(image):
    if image is None:
        return "", None

    try:
        # Gradio gives RGB image
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detect hand
        hands = hd.findHands(frame, draw=False, flipType=True)

        if not hands:
            return "No hand detected", image

        hand = hands[0]
        x, y, w, h = hand["bbox"]

        offset = 29

        # Keep coordinates inside image
        y1 = max(0, y - offset)
        y2 = min(frame.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(frame.shape[1], x + w + offset)

        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            return "No hand detected", image

        # Detect landmarks on cropped hand
        handz = hd.findHands(cropped, draw=False, flipType=True)

        if not handz:
            return "No hand detected", image

        hand = handz[0]
        pts = hand["lmList"]

        # Create white 400x400 skeleton image
        white = np.ones((400, 400, 3), dtype=np.uint8) * 255

        # Center hand landmarks
        os = ((400 - w) // 2) - 15
        os1 = ((400 - h) // 2) - 15

        # Draw hand skeleton
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
            (0, 5), (0, 17)
        ]

        for a, b in connections:
            cv2.line(
                white,
                (pts[a][0] + os, pts[a][1] + os1),
                (pts[b][0] + os, pts[b][1] + os1),
                (0, 255, 0),
                3
            )

        for i in range(21):
            cv2.circle(
                white,
                (pts[i][0] + os, pts[i][1] + os1),
                2,
                (0, 0, 255),
                1
            )

        # Model prediction
        model_input = white.reshape(1, 400, 400, 3)
        prob = np.array(model.predict(model_input, verbose=0)[0], dtype="float32")

        ch1 = np.argmax(prob)
        prob[ch1] = 0

        ch2 = np.argmax(prob)

        # Basic class mapping used by your project
        groups = {
            0: "A",
            1: "B",
            2: "C",
            3: "G",
            4: "L",
            5: "P",
            6: "X",
            7: "Y"
        }

        prediction = groups.get(int(ch1), str(ch1))

        # Show prediction on image
        output = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)

        return prediction, output

    except Exception as e:
        return f"Error: {str(e)}", image


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", sources=["webcam", "upload"]),
    outputs=[
        gr.Textbox(label="Predicted Sign"),
        gr.Image(label="Processed Hand")
    ],
    title="GestureSpeak - Sign Language to Text",
    description="Show a hand sign using your webcam and get the predicted sign."
)

if __name__ == "__main__":
    demo.launch()