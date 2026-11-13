# HealthPlain AI — Plain-Language Healthcare Intelligence

HealthPlain AI is a modern healthcare document intelligence and clinical workflow platform built with Flask, SQLite, and Machine Learning.

## Features
- **Direct Role Access (No Login / Sign Up / OTP)**: Choose directly between **Patient**, **Doctor**, and **Hospital Admin** from the Home page.
- **Patient Dashboard (6 Tabs)**: Dashboard, Report Simplifier, Medicine Reminders, Food & Wellness, Find Doctors, Recovery Check-in.
- **Doctor Dashboard (4 Tabs)**: Dashboard, Appointments, Prescription, Recovery Check-ins.
- **Hospital Admin Dashboard (4 Tabs)**: Dashboard, Doctors, Patients, Publish Slots.
- **Full English & Telugu (తెలుగు) Translation**: Instant language switch in the top-right corner across all roles and tabs.
- **Tuscan Sunset Theme**: Coral (`#E35336`), Peach (`#FFD3AC`), Lavender (`#9988A1`), and Deep Brown (`#8A2B0E`).

## How to Run
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```
Open `http://127.0.0.1:5000` in your browser.
