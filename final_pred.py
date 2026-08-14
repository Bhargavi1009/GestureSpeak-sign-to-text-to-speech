# Importing Libraries
import numpy as np
import math
import cv2
import os
import sys
import traceback
import pyttsx3

from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant

import tkinter as tk
from PIL import Image, ImageTk


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

ddd = enchant.Dict("en-US")

hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


# ---------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------

class Application:

    def __init__(self):

        # Camera
        self.vs = cv2.VideoCapture(0)
        self.current_image = None

        # Model
        self.model = load_model("cnn8grps_rad1_model.h5")

        # Speech
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)

        voices = self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice", voices[0].id)

        # Character variables
        self.ct = {}
        self.ct["blank"] = 0

        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""

        self.count = -1

        self.ten_prev_char = []
        for i in range(10):
            self.ten_prev_char.append(" ")

        for i in ascii_uppercase:
            self.ct[i] = 0

        print("Loaded model from disk")

        # -------------------------------------------------
        # MAIN WINDOW
        # -------------------------------------------------

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)

        # Window size
        self.root.geometry("1300x750")
        self.root.resizable(False, False)

        # -------------------------------------------------
        # TITLE
        # -------------------------------------------------

        self.T = tk.Label(
            self.root,
            text="Sign Language To Text Conversion",
            font=("Courier", 28, "bold")
        )
        self.T.place(x=330, y=15)

        # -------------------------------------------------
        # CAMERA PANEL
        # -------------------------------------------------

        self.panel = tk.Label(self.root)
        self.panel.place(
            x=30,
            y=80,
            width=520,
            height=400
        )

        # -------------------------------------------------
        # HAND LANDMARK PANEL
        # -------------------------------------------------

        self.panel2 = tk.Label(self.root)
        self.panel2.place(
            x=650,
            y=80,
            width=400,
            height=400
        )

        # -------------------------------------------------
        # CURRENT CHARACTER
        # -------------------------------------------------

        self.T1 = tk.Label(
            self.root,
            text="Character :",
            font=("Courier", 24, "bold")
        )
        self.T1.place(x=30, y=500)

        self.panel3 = tk.Label(
            self.root,
            text="",
            font=("Courier", 24, "bold")
        )
        self.panel3.place(x=250, y=500)

        # -------------------------------------------------
        # SENTENCE
        # -------------------------------------------------

        self.T3 = tk.Label(
            self.root,
            text="Sentence :",
            font=("Courier", 24, "bold")
        )
        self.T3.place(x=30, y=545)

        self.panel5 = tk.Label(
            self.root,
            text="",
            font=("Courier", 22, "bold"),
            anchor="w",
            justify="left"
        )
        self.panel5.place(
            x=250,
            y=545,
            width=1000,
            height=45
        )

        # -------------------------------------------------
        # SUGGESTIONS LABEL
        # -------------------------------------------------

        self.T4 = tk.Label(
            self.root,
            text="Suggestions :",
            fg="red",
            font=("Courier", 20, "bold")
        )
        self.T4.place(x=30, y=610)

        # -------------------------------------------------
        # SUGGESTION BUTTONS
        # -------------------------------------------------

        # Button 1
        self.b1 = tk.Button(
            self.root,
            text="",
            font=("Courier", 16),
            width=18,
            height=1,
            command=self.action1
        )
        self.b1.place(x=220, y=605)

        # Button 2
        self.b2 = tk.Button(
            self.root,
            text="",
            font=("Courier", 16),
            width=18,
            height=1,
            command=self.action2
        )
        self.b2.place(x=440, y=605)

        # Button 3
        self.b3 = tk.Button(
            self.root,
            text="",
            font=("Courier", 16),
            width=18,
            height=1,
            command=self.action3
        )
        self.b3.place(x=660, y=605)

        # Button 4
        self.b4 = tk.Button(
            self.root,
            text="",
            font=("Courier", 16),
            width=18,
            height=1,
            command=self.action4
        )
        self.b4.place(x=880, y=605)

        # -------------------------------------------------
        # CLEAR BUTTON
        # -------------------------------------------------

        self.clear = tk.Button(
            self.root,
            text="Clear",
            font=("Courier", 18, "bold"),
            width=9,
            height=2,
            command=self.clear_fun
        )
        self.clear.place(x=1010, y=680)

        # -------------------------------------------------
        # SPEAK BUTTON
        # -------------------------------------------------

        self.speak = tk.Button(
            self.root,
            text="Speak",
            font=("Courier", 18, "bold"),
            width=9,
            height=2,
            command=self.speak_fun
        )
        self.speak.place(x=1150, y=680)

        # -------------------------------------------------
        # VARIABLES
        # -------------------------------------------------

        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"

        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        # Start camera loop
        self.video_loop()

    # =====================================================
    # VIDEO LOOP
    # =====================================================

    def video_loop(self):

        try:

            ok, frame = self.vs.read()

            if not ok:
                self.root.after(10, self.video_loop)
                return

            cv2image = cv2.flip(frame, 1)

            hands = hd.findHands(
                cv2image,
                draw=False,
                flipType=True
            )

            cv2image_copy = np.array(cv2image)

            cv2image = cv2.cvtColor(
                cv2image,
                cv2.COLOR_BGR2RGB
            )

            self.current_image = Image.fromarray(cv2image)

            imgtk = ImageTk.PhotoImage(
                image=self.current_image
            )

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # -------------------------------------------------
            # HAND FOUND
            # -------------------------------------------------

            if hands:

                hand = hands[0]

                x, y, w, h = hand["bbox"]

                # Prevent negative crop coordinates
                x1 = max(0, x - offset)
                y1 = max(0, y - offset)

                x2 = min(
                    cv2image_copy.shape[1],
                    x + w + offset
                )

                y2 = min(
                    cv2image_copy.shape[0],
                    y + h + offset
                )

                image = cv2image_copy[y1:y2, x1:x2]

                if image.size > 0:

                    handz = hd2.findHands(
                        image,
                        draw=False,
                        flipType=True
                    )

                    self.ccc += 1

                    if handz:

                        hand = handz[0]

                        self.pts = hand["lmList"]

                        # -------------------------------------------------
                        # WHITE IMAGE
                        # -------------------------------------------------

                        white = np.ones(
                            (400, 400, 3),
                            dtype=np.uint8
                        ) * 255

                        os_x = ((400 - w) // 2) - 15
                        os_y = ((400 - h) // 2) - 15

                        # -------------------------------------------------
                        # DRAW HAND
                        # -------------------------------------------------

                        for t in range(0, 4):

                            cv2.line(
                                white,
                                (
                                    self.pts[t][0] + os_x,
                                    self.pts[t][1] + os_y
                                ),
                                (
                                    self.pts[t + 1][0] + os_x,
                                    self.pts[t + 1][1] + os_y
                                ),
                                (0, 255, 0),
                                3
                            )

                        for t in range(5, 8):

                            cv2.line(
                                white,
                                (
                                    self.pts[t][0] + os_x,
                                    self.pts[t][1] + os_y
                                ),
                                (
                                    self.pts[t + 1][0] + os_x,
                                    self.pts[t + 1][1] + os_y
                                ),
                                (0, 255, 0),
                                3
                            )

                        for t in range(9, 12):

                            cv2.line(
                                white,
                                (
                                    self.pts[t][0] + os_x,
                                    self.pts[t][1] + os_y
                                ),
                                (
                                    self.pts[t + 1][0] + os_x,
                                    self.pts[t + 1][1] + os_y
                                ),
                                (0, 255, 0),
                                3
                            )

                        for t in range(13, 16):

                            cv2.line(
                                white,
                                (
                                    self.pts[t][0] + os_x,
                                    self.pts[t][1] + os_y
                                ),
                                (
                                    self.pts[t + 1][0] + os_x,
                                    self.pts[t + 1][1] + os_y
                                ),
                                (0, 255, 0),
                                3
                            )

                        for t in range(17, 20):

                            cv2.line(
                                white,
                                (
                                    self.pts[t][0] + os_x,
                                    self.pts[t][1] + os_y
                                ),
                                (
                                    self.pts[t + 1][0] + os_x,
                                    self.pts[t + 1][1] + os_y
                                ),
                                (0, 255, 0),
                                3
                            )

                        # Palm lines

                        palm_lines = [
                            (5, 9),
                            (9, 13),
                            (13, 17),
                            (0, 5),
                            (0, 17)
                        ]

                        for a, b in palm_lines:

                            cv2.line(
                                white,
                                (
                                    self.pts[a][0] + os_x,
                                    self.pts[a][1] + os_y
                                ),
                                (
                                    self.pts[b][0] + os_x,
                                    self.pts[b][1] + os_y
                                ),
                                (0, 255, 0),
                                3
                            )

                        # Points

                        for i in range(21):

                            cv2.circle(
                                white,
                                (
                                    self.pts[i][0] + os_x,
                                    self.pts[i][1] + os_y
                                ),
                                2,
                                (0, 0, 255),
                                1
                            )

                        # -------------------------------------------------
                        # PREDICT
                        # -------------------------------------------------

                        res = white

                        self.predict(res)

                        # -------------------------------------------------
                        # SHOW HAND IMAGE
                        # -------------------------------------------------

                        self.current_image2 = Image.fromarray(res)

                        imgtk2 = ImageTk.PhotoImage(
                            image=self.current_image2
                        )

                        self.panel2.imgtk = imgtk2
                        self.panel2.config(image=imgtk2)

                        # Current character

                        self.panel3.config(
                            text=self.current_symbol,
                            font=("Courier", 24, "bold")
                        )

                        # Suggestions

                        self.b1.config(
                            text=self.word1,
                            font=("Courier", 16),
                            command=self.action1
                        )

                        self.b2.config(
                            text=self.word2,
                            font=("Courier", 16),
                            command=self.action2
                        )

                        self.b3.config(
                            text=self.word3,
                            font=("Courier", 16),
                            command=self.action3
                        )

                        self.b4.config(
                            text=self.word4,
                            font=("Courier", 16),
                            command=self.action4
                        )

            # Sentence

            self.panel5.config(
                text=self.str,
                font=("Courier", 22, "bold"),
                wraplength=1000
            )

        except Exception:

            print(traceback.format_exc())

        finally:

            self.root.after(
                10,
                self.video_loop
            )

    # =====================================================
    # DISTANCE
    # =====================================================

    def distance(self, x, y):

        return math.sqrt(
            ((x[0] - y[0]) ** 2) +
            ((x[1] - y[1]) ** 2)
        )

    # =====================================================
    # SUGGESTION ACTIONS
    # =====================================================

    def action1(self):

        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(
            self.word,
            idx_space
        )

        if idx_word >= 0:

            self.str = self.str[:idx_word]
            self.str += self.word1.upper()

    def action2(self):

        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(
            self.word,
            idx_space
        )

        if idx_word >= 0:

            self.str = self.str[:idx_word]
            self.str += self.word2.upper()

    def action3(self):

        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(
            self.word,
            idx_space
        )

        if idx_word >= 0:

            self.str = self.str[:idx_word]
            self.str += self.word3.upper()

    def action4(self):

        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(
            self.word,
            idx_space
        )

        if idx_word >= 0:

            self.str = self.str[:idx_word]
            self.str += self.word4.upper()

    # =====================================================
    # SPEAK
    # =====================================================

    def speak_fun(self):

        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()

    # =====================================================
    # CLEAR
    # =====================================================

    def clear_fun(self):

        self.str = " "

        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, test_image):

        white = test_image

        white = white.reshape(
            1,
            400,
            400,
            3
        )

        prob = np.array(
            self.model.predict(white, verbose=0)[0],
            dtype="float32"
        )

        ch1 = np.argmax(prob, axis=0)

        prob[ch1] = 0

        ch2 = np.argmax(prob, axis=0)

        prob[ch2] = 0

        ch3 = np.argmax(prob, axis=0)

        prob[ch3] = 0

        pl = [ch1, ch2]

        # -------------------------------------------------
        # CONDITION FOR [AEMNST]
        # -------------------------------------------------

        l = [
            [5, 2], [5, 3], [3, 5], [3, 6],
            [3, 0], [3, 2], [6, 4], [6, 1],
            [6, 2], [6, 6], [6, 7], [6, 0],
            [6, 5], [4, 1], [1, 0], [1, 1],
            [6, 3], [1, 6], [5, 6], [5, 1],
            [4, 5], [1, 4], [1, 5], [2, 0],
            [2, 6], [4, 6], [1, 0], [5, 7],
            [1, 6], [6, 1], [7, 6], [2, 5],
            [7, 1], [5, 4], [7, 0], [7, 5],
            [7, 2]
        ]

        if pl in l:

            if (
                self.pts[6][1] < self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
            ):
                ch1 = 0

        # -------------------------------------------------
        # CONDITION FOR O / S
        # -------------------------------------------------

        l = [[2, 2], [2, 1]]

        if pl in l:

            if self.pts[5][0] < self.pts[4][0]:

                ch1 = 0

        # -------------------------------------------------
        # C / AEMNST
        # -------------------------------------------------

        l = [
            [0, 0], [0, 6], [0, 2],
            [0, 5], [0, 1], [0, 7],
            [5, 2], [7, 6], [7, 1]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[0][0] > self.pts[8][0]
                and self.pts[0][0] > self.pts[4][0]
                and self.pts[0][0] > self.pts[12][0]
                and self.pts[0][0] > self.pts[16][0]
                and self.pts[0][0] > self.pts[20][0]
                and self.pts[5][0] > self.pts[4][0]
            ):
                ch1 = 2

        # -------------------------------------------------
        # C / O
        # -------------------------------------------------

        l = [[6, 0], [6, 6], [6, 2]]

        pl = [ch1, ch2]

        if pl in l:

            if self.distance(
                self.pts[8],
                self.pts[16]
            ) < 52:

                ch1 = 2

        # -------------------------------------------------
        # G / H
        # -------------------------------------------------

        l = [
            [1, 4], [1, 5],
            [1, 6], [1, 3],
            [1, 0]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
                and self.pts[0][0] < self.pts[8][0]
                and self.pts[0][0] < self.pts[12][0]
                and self.pts[0][0] < self.pts[16][0]
                and self.pts[0][0] < self.pts[20][0]
            ):

                ch1 = 3

        # -------------------------------------------------
        # G / L
        # -------------------------------------------------

        l = [
            [4, 6], [4, 1],
            [4, 5], [4, 3],
            [4, 7]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[4][0] > self.pts[0][0]:

                ch1 = 3

        # -------------------------------------------------
        # G / PQZ
        # -------------------------------------------------

        l = [
            [5, 3], [5, 0],
            [5, 7], [5, 4],
            [5, 2], [5, 1],
            [5, 5]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[2][1] + 15 < self.pts[16][1]:

                ch1 = 3

        # -------------------------------------------------
        # L / X
        # -------------------------------------------------

        l = [[6, 4], [6, 1], [6, 2]]

        pl = [ch1, ch2]

        if pl in l:

            if self.distance(
                self.pts[4],
                self.pts[11]
            ) > 55:

                ch1 = 4

        # -------------------------------------------------
        # L / D
        # -------------------------------------------------

        l = [[1, 4], [1, 6], [1, 1]]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.distance(
                    self.pts[4],
                    self.pts[11]
                ) > 50
                and self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
            ):

                ch1 = 4

        # -------------------------------------------------
        # L / GH
        # -------------------------------------------------

        l = [[3, 6], [3, 4]]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[4][0] < self.pts[0][0]:

                ch1 = 4

        # -------------------------------------------------
        # L / C0
        # -------------------------------------------------

        l = [[2, 2], [2, 5], [2, 4]]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[1][0] < self.pts[12][0]:

                ch1 = 4

        # -------------------------------------------------
        # GH / Z
        # -------------------------------------------------

        l = [[3, 6], [3, 5], [3, 4]]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
                and self.pts[4][1] > self.pts[10][1]
            ):

                ch1 = 5

        # -------------------------------------------------
        # GH / PQ
        # -------------------------------------------------

        l = [[3, 2], [3, 1], [3, 6]]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[4][1] + 17 > self.pts[8][1]
                and self.pts[4][1] + 17 > self.pts[12][1]
                and self.pts[4][1] + 17 > self.pts[16][1]
                and self.pts[4][1] + 17 > self.pts[20][1]
            ):

                ch1 = 5

        # -------------------------------------------------
        # L / PQZ
        # -------------------------------------------------

        l = [
            [4, 4], [4, 5],
            [4, 2], [7, 5],
            [7, 6], [7, 0]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[4][0] > self.pts[0][0]:

                ch1 = 5

        # -------------------------------------------------
        # PQZ / AEMNST
        # -------------------------------------------------

        l = [
            [0, 2], [0, 6], [0, 1],
            [0, 5], [0, 0], [0, 7],
            [0, 4], [0, 3], [2, 7]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[0][0] < self.pts[8][0]
                and self.pts[0][0] < self.pts[12][0]
                and self.pts[0][0] < self.pts[16][0]
                and self.pts[0][0] < self.pts[20][0]
            ):

                ch1 = 5

        # -------------------------------------------------
        # PQZ / YJ
        # -------------------------------------------------

        l = [[5, 7], [5, 2], [5, 6]]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[3][0] < self.pts[0][0]:

                ch1 = 7

        # -------------------------------------------------
        # L / YJ
        # -------------------------------------------------

        l = [
            [4, 6], [4, 2],
            [4, 4], [4, 1],
            [4, 5], [4, 7]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[6][1] < self.pts[8][1]:

                ch1 = 7

        # -------------------------------------------------
        # X / YJ
        # -------------------------------------------------

        l = [
            [6, 7], [0, 7],
            [0, 1], [0, 0],
            [6, 4], [6, 6],
            [6, 5], [6, 1]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[18][1] > self.pts[20][1]:

                ch1 = 7

        # -------------------------------------------------
        # X / AEMNST
        # -------------------------------------------------

        l = [
            [0, 4], [0, 2],
            [0, 3], [0, 1],
            [0, 6]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[5][0] > self.pts[16][0]:

                ch1 = 6

        # -------------------------------------------------
        # YJ / X
        # -------------------------------------------------

        l = [[7, 2]]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[18][1] < self.pts[20][1]
                and self.pts[8][1] < self.pts[10][1]
            ):

                ch1 = 6

        # -------------------------------------------------
        # C0 / X
        # -------------------------------------------------

        l = [
            [2, 1], [2, 2],
            [2, 6], [2, 7],
            [2, 0]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if self.distance(
                self.pts[8],
                self.pts[16]
            ) > 50:

                ch1 = 6

        # -------------------------------------------------
        # L / X
        # -------------------------------------------------

        l = [
            [4, 6], [4, 2],
            [4, 1], [4, 4]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if self.distance(
                self.pts[4],
                self.pts[11]
            ) < 60:

                ch1 = 6

        # -------------------------------------------------
        # X / D
        # -------------------------------------------------

        l = [
            [1, 4], [1, 6],
            [1, 0], [1, 2]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[5][0] - self.pts[4][0] - 15 > 0:

                ch1 = 6

        # -------------------------------------------------
        # B / PQZ
        # -------------------------------------------------

        l = [
            [5, 0], [5, 1],
            [5, 4], [5, 5],
            [5, 6], [6, 1],
            [7, 6], [0, 2],
            [7, 1], [7, 4],
            [6, 6], [7, 2],
            [6, 3], [6, 4],
            [7, 5]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] > self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):

                ch1 = 1

        # -------------------------------------------------
        # F / PQZ
        # -------------------------------------------------

        l = [
            [6, 1], [6, 0],
            [0, 3], [6, 4],
            [2, 2], [0, 6],
            [6, 2], [7, 6],
            [4, 6], [4, 1],
            [4, 2], [0, 2],
            [7, 1], [7, 4],
            [6, 6], [7, 2],
            [7, 5]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[6][1] < self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] > self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):

                ch1 = 1

        # -------------------------------------------------
        # F GROUP
        # -------------------------------------------------

        l = [
            [6, 1], [6, 0],
            [4, 2], [4, 1],
            [4, 6], [4, 4]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] > self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):

                ch1 = 1

        # -------------------------------------------------
        # D / PQZ
        # -------------------------------------------------

        l = [
            [5, 0], [3, 4],
            [3, 0], [3, 1],
            [3, 5], [5, 5],
            [5, 4], [5, 1],
            [7, 6]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
                and self.pts[2][0] < self.pts[0][0]
                and self.pts[4][1] > self.pts[14][1]
            ):

                ch1 = 1

        # -------------------------------------------------
        # D GROUP
        # -------------------------------------------------

        l = [[4, 1], [4, 2], [4, 4]]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.distance(
                    self.pts[4],
                    self.pts[11]
                ) < 50
                and self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
            ):

                ch1 = 1

        # -------------------------------------------------
        # D GROUP 2
        # -------------------------------------------------

        l = [
            [3, 4], [3, 0],
            [3, 1], [3, 5],
            [3, 6]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
                and self.pts[2][0] < self.pts[0][0]
                and self.pts[14][1] < self.pts[4][1]
            ):

                ch1 = 1

        # -------------------------------------------------
        # D / L
        # -------------------------------------------------

        l = [
            [6, 6], [6, 4],
            [6, 1], [6, 2]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if self.pts[5][0] - self.pts[4][0] - 15 < 0:

                ch1 = 1

        # -------------------------------------------------
        # I / PQZ
        # -------------------------------------------------

        l = [
            [5, 4], [5, 5],
            [5, 1], [0, 3],
            [0, 7], [5, 0],
            [0, 2], [6, 2],
            [7, 5], [7, 1],
            [7, 6], [7, 7]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[6][1] < self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):

                ch1 = 1

        # -------------------------------------------------
        # YJ / BFDI
        # -------------------------------------------------

        l = [
            [1, 5], [1, 7],
            [1, 1], [1, 6],
            [1, 3], [1, 0]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[4][0] < self.pts[5][0] + 15
                and self.pts[6][1] < self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):

                ch1 = 7

        # -------------------------------------------------
        # UVR
        # -------------------------------------------------

        l = [
            [5, 5], [5, 0],
            [5, 4], [5, 1],
            [4, 6], [4, 1],
            [7, 6], [3, 0],
            [3, 5]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
                and self.pts[4][1] > self.pts[14][1]
            ):

                ch1 = 1

        # -------------------------------------------------
        # W
        # -------------------------------------------------

        fg = 13

        l = [
            [3, 5], [3, 0],
            [3, 6], [5, 1],
            [4, 1], [2, 0],
            [5, 0], [5, 5]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                not (
                    self.pts[0][0] + fg < self.pts[8][0]
                    and self.pts[0][0] + fg < self.pts[12][0]
                    and self.pts[0][0] + fg < self.pts[16][0]
                    and self.pts[0][0] + fg < self.pts[20][0]
                )
                and not (
                    self.pts[0][0] > self.pts[8][0]
                    and self.pts[0][0] > self.pts[12][0]
                    and self.pts[0][0] > self.pts[16][0]
                    and self.pts[0][0] > self.pts[20][0]
                )
                and self.distance(
                    self.pts[4],
                    self.pts[11]
                ) < 50
            ):

                ch1 = 1

        l = [
            [5, 0],
            [5, 5],
            [0, 1]
        ]

        pl = [ch1, ch2]

        if pl in l:

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] > self.pts[16][1]
            ):

                ch1 = 1

        # =================================================
        # SUBGROUP CONDITIONS
        # =================================================

        if ch1 == 0:

            ch1 = "S"

            if (
                self.pts[4][0] < self.pts[6][0]
                and self.pts[4][0] < self.pts[10][0]
                and self.pts[4][0] < self.pts[14][0]
                and self.pts[4][0] < self.pts[18][0]
            ):

                ch1 = "A"

            if (
                self.pts[4][0] > self.pts[6][0]
                and self.pts[4][0] < self.pts[10][0]
                and self.pts[4][0] < self.pts[14][0]
                and self.pts[4][0] < self.pts[18][0]
                and self.pts[4][1] < self.pts[14][1]
                and self.pts[4][1] < self.pts[18][1]
            ):

                ch1 = "T"

            if (
                self.pts[4][1] > self.pts[8][1]
                and self.pts[4][1] > self.pts[12][1]
                and self.pts[4][1] > self.pts[16][1]
                and self.pts[4][1] > self.pts[20][1]
            ):

                ch1 = "E"

            if (
                self.pts[4][0] > self.pts[6][0]
                and self.pts[4][0] > self.pts[10][0]
                and self.pts[4][0] > self.pts[14][0]
                and self.pts[4][1] < self.pts[18][1]
            ):

                ch1 = "M"

            if (
                self.pts[4][0] > self.pts[6][0]
                and self.pts[4][0] > self.pts[10][0]
                and self.pts[4][1] < self.pts[18][1]
                and self.pts[4][1] < self.pts[14][1]
            ):

                ch1 = "N"

        if ch1 == 2:

            if self.distance(
                self.pts[12],
                self.pts[4]
            ) > 42:

                ch1 = "C"

            else:

                ch1 = "O"

        if ch1 == 3:

            if self.distance(
                self.pts[8],
                self.pts[12]
            ) > 72:

                ch1 = "G"

            else:

                ch1 = "H"

        if ch1 == 7:

            if self.distance(
                self.pts[8],
                self.pts[4]
            ) > 42:

                ch1 = "Y"

            else:

                ch1 = "J"

        if ch1 == 4:

            ch1 = "L"

        if ch1 == 6:

            ch1 = "X"

        if ch1 == 5:

            if (
                self.pts[4][0] > self.pts[12][0]
                and self.pts[4][0] > self.pts[16][0]
                and self.pts[4][0] > self.pts[20][0]
            ):

                if self.pts[8][1] < self.pts[5][1]:

                    ch1 = "Z"

                else:

                    ch1 = "Q"

            else:

                ch1 = "P"

        # -------------------------------------------------
        # B D F I W K U V R
        # -------------------------------------------------

        if ch1 == 1:

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] > self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):

                ch1 = "B"

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
            ):

                ch1 = "D"

            if (
                self.pts[6][1] < self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] > self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):

                ch1 = "F"

            if (
                self.pts[6][1] < self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):

                ch1 = "I"

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] > self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
            ):

                ch1 = "W"

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
                and self.pts[4][1] < self.pts[9][1]
            ):

                ch1 = "K"

            if (
                (
                    self.distance(
                        self.pts[8],
                        self.pts[12]
                    )
                    -
                    self.distance(
                        self.pts[6],
                        self.pts[10]
                    )
                ) < 8
                and self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
            ):

                ch1 = "U"

            if (
                (
                    self.distance(
                        self.pts[8],
                        self.pts[12]
                    )
                    -
                    self.distance(
                        self.pts[6],
                        self.pts[10]
                    )
                ) >= 8
                and self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
                and self.pts[4][1] > self.pts[9][1]
            ):

                ch1 = "V"

            if (
                self.pts[8][0] > self.pts[12][0]
                and self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] < self.pts[20][1]
            ):

                ch1 = "R"

        # -------------------------------------------------
        # SPACE
        # -------------------------------------------------

        if (
            ch1 == 1
            or ch1 == "E"
            or ch1 == "S"
            or ch1 == "X"
            or ch1 == "Y"
            or ch1 == "B"
        ):

            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):

                ch1 = " "

        # -------------------------------------------------
        # NEXT
        # -------------------------------------------------

        if ch1 in ["E", "Y", "B"]:

            if (
                self.pts[4][0] < self.pts[5][0]
                and self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] > self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):

                ch1 = "next"

        # -------------------------------------------------
        # BACKSPACE
        # -------------------------------------------------

        if (
            self.pts[0][0] > self.pts[8][0]
            and self.pts[0][0] > self.pts[12][0]
            and self.pts[0][0] > self.pts[16][0]
            and self.pts[0][0] > self.pts[20][0]
            and self.pts[4][1] < self.pts[8][1]
            and self.pts[4][1] < self.pts[12][1]
            and self.pts[4][1] < self.pts[16][1]
            and self.pts[4][1] < self.pts[20][1]
            and self.pts[4][1] < self.pts[6][1]
            and self.pts[4][1] < self.pts[10][1]
            and self.pts[4][1] < self.pts[14][1]
            and self.pts[4][1] < self.pts[18][1]
        ):

            ch1 = "Backspace"

        # -------------------------------------------------
        # NEXT ACTION
        # -------------------------------------------------

        if ch1 == "next" and self.prev_char != "next":

            previous = self.ten_prev_char[
                (self.count - 2) % 10
            ]

            if previous == "Backspace":

                self.str = self.str[:-1]

            elif previous != "next":

                self.str += previous

        # -------------------------------------------------
        # SPACE
        # -------------------------------------------------

        if ch1 == "  " and self.prev_char != "  ":

            self.str += "  "

        # -------------------------------------------------
        # SAVE CHARACTER
        # -------------------------------------------------

        self.prev_char = ch1

        self.current_symbol = ch1

        self.count += 1

        self.ten_prev_char[
            self.count % 10
        ] = ch1

        # -------------------------------------------------
        # WORD SUGGESTIONS
        # -------------------------------------------------

        if len(self.str.strip()) != 0:

            st = self.str.rfind(" ")
            ed = len(self.str)

            word = self.str[
                st + 1:ed
            ]

            self.word = word

            if len(word.strip()) != 0:

                suggestions = ddd.suggest(word)

                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "

                if len(suggestions) >= 1:

                    self.word1 = suggestions[0]

                if len(suggestions) >= 2:

                    self.word2 = suggestions[1]

                if len(suggestions) >= 3:

                    self.word3 = suggestions[2]

                if len(suggestions) >= 4:

                    self.word4 = suggestions[3]

            else:

                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "

    # =====================================================
    # DESTROY APPLICATION
    # =====================================================

    def destructor(self):

        print(self.ten_prev_char)

        self.root.destroy()

        self.vs.release()

        cv2.destroyAllWindows()


# =========================================================
# START APPLICATION
# =========================================================

print("Starting Application...")

Application().root.mainloop()
