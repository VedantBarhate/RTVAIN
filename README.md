# ⚽🎙️ RTVAIN

### An AI that watches football penalties and commentates them live

RTVAIN watches a live football penalty feed, understands what is happening, and narrates it in real time, like a real commentator. It combines computer vision, event-detection logic, an LLM and text-to-speech into one pipeline.

> Born mid-game while playing FIFA 08. Built as a Semester 5 PBL project.

---

## 📖 Overview

The idea started with one question: *what if an AI could watch me play and commentate like a real broadcaster?*

Commentating on a full match needed more compute than was available, so the scope was narrowed to **football penalty kicks**. That domain is small enough to run on a laptop and still rich enough to tell a story (kick, save, goal, miss).

RTVAIN is not a single model. It is a **full real-time pipeline**: perception → understanding → language → speech.

---

## 🧠 How It Works

```
 Live Video Feed
        │
        ▼
┌───────────────────┐
│  YOLO Detection   │  Ball · Kicker · Goalkeeper · Goalpost
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Context Engine   │  Tracks movement → Kick? Goal? Save? Miss?
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Event Sequence   │  Structured stream of match events
└─────────┬─────────┘
          ▼
┌───────────────────┐
│       LLM         │  Turns events into live commentary text
└─────────┬─────────┘
          ▼
┌───────────────────┐
│       TTS         │  Speaks it like a commentator 🎙️
└───────────────────┘
```

| Stage | What it does |
|---|---|
| **1. Object Detection (YOLO)** | Detects the ball, kicker, goalkeeper and goalpost in each frame of the live feed. |
| **2. Context Engine** | Analyses object movement across frames to infer events: **kick**, **goal**, **save**, **miss**. |
| **3. Event Sequence** | Detected events are ordered into a sequence that gives the LLM context. |
| **4. LLM Commentary** | The LLM converts the event sequence into natural, broadcaster-style commentary. |
| **5. Text-to-Speech** | The commentary is spoken aloud in real time. |

---

## ✨ Features

- 🎯 Real-time detection of ball, kicker, goalkeeper and goalpost
- 🧩 Rule-based context engine that recognises penalty events (kick / goal / save / miss)
- 🗣️ LLM-generated live commentary
- 🔊 Spoken output through TTS
- 💻 Designed to run on modest hardware

---

## 🛠️ Tech Stack

<!-- TODO: Confirm and edit this list to match the repo -->

| Component | Technology |
|---|---|
| Object Detection | YOLO (YOLOv11) |
| Video Processing | OpenCV |
| LLM | Groq API |
| Text-to-Speech | Edge TTS |
| Language | Python |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A webcam, screen capture or video file as the input feed


### Installation

```bash
git clone https://github.com/VedantBarhate/RTVAIN.git
cd RTVAIN
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

---

## 📊 Dataset & Training

The detection model was trained on a custom dataset of penalty-kick footage, annotated with four classes:

1. Ball
2. Kicker
3. Goalkeeper
4. Goalpost

---

## ⚠️ Limitations

Honest notes on where it stands:

- Scope is limited to **penalty kicks**, not full matches
- Event detection is heuristic-based and can misfire in edge cases
- Latency depends on hardware, LLM choice and TTS engine
- Not perfect and not magical, but real, functioning and satisfying

---

## 🔮 Future Work

- Extend to full-match commentary
- Add more events (fouls, corners, free kicks)
- Support multiple commentary styles and languages
- Reduce end-to-end latency
- Add player and team recognition

---

## 👥 Team

Built as a Semester 5 PBL project by:

- **Vedant Barhate** (project idea, system architecture and pipeline)
- **[Achyut Yesare](https://www.linkedin.com/in/achyut-yesare-a5754a31b/)** (dataset, annotation, reporting, testing)
- **[Yash Pandey](https://www.linkedin.com/in/yash-pandey-151961213/)** (dataset, annotation, reporting, testing)
- **[Vibhor Dev](https://www.linkedin.com/in/vibhor-dev-9153aa276/)** (dataset, annotation, reporting, testing)

---

## 💡 What I Learned

> Sometimes you don't start a project because you already know how. You start it because it feels interesting enough to figure out along the way.

If your idea feels "too big", just start anyway. You'll learn while doing.
