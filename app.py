import os
import re
import uuid
import sqlite3
import json
import secrets
from datetime import datetime
from functools import wraps
from pathlib import Path

import joblib
import fitz
import pytesseract
from PIL import Image
from docx import Document
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from i18n import get_text

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "document_classifier.joblib"
DB_PATH = BASE_DIR / "healthplain.db"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "txt", "docx"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.jinja_env.globals.update(get_text=get_text)
app.secret_key = os.environ.get("HEALTHPLAIN_SECRET", "HEALTHPLAIN-SECURE-DEFAULT-KEY-2026")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

UPLOAD_DIR.mkdir(exist_ok=True)
(MODEL_PATH.parent).mkdir(exist_ok=True)


@app.before_request
def csrf_protect():
    if request.method == "POST":
        token = session.get("csrf_token")
        if not token or token != request.form.get("csrf_token"):
            return "CSRF Token Missing or Invalid", 400


@app.context_processor
def inject_globals():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return dict(
        csrf_token=lambda: session["csrf_token"],
        current_lang=session.get("lang", "en"),
        current_role=session.get("role")
    )


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT,
            password_hash TEXT NOT NULL DEFAULT 'direct-access',
            role TEXT NOT NULL DEFAULT 'patient',
            verified INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            confidence REAL NOT NULL,
            extracted_text TEXT,
            extracted_values TEXT,
            status TEXT NOT NULL,
            doctor_notes TEXT,
            created_at TEXT NOT NULL
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_name TEXT NOT NULL,
            dosage TEXT NOT NULL,
            schedule_time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending',
            created_at TEXT NOT NULL
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS recovery_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL DEFAULT 'Ananya Rao',
            wellbeing TEXT NOT NULL,
            pain_level INTEGER NOT NULL DEFAULT 2,
            notes TEXT,
            created_at TEXT NOT NULL
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            experience INTEGER NOT NULL DEFAULT 8,
            verified INTEGER NOT NULL DEFAULT 1,
            photo_url TEXT
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS patients_directory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            phone TEXT,
            condition_summary TEXT,
            created_at TEXT NOT NULL
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS appointment_slots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER NOT NULL,
            doctor_name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            slot_date TEXT NOT NULL,
            slot_time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Available',
            patient_name TEXT,
            reason TEXT
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            medicine_name TEXT NOT NULL,
            dosage TEXT NOT NULL,
            instructions TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")

        try:
            conn.execute("ALTER TABLE reports ADD COLUMN doctor_notes TEXT")
        except sqlite3.OperationalError:
            pass

        existing_user = conn.execute("SELECT id FROM users LIMIT 1").fetchone()
        if not existing_user:
            conn.execute(
                "INSERT INTO users (id, name, email, phone, password_hash, role, verified, created_at) VALUES (1, 'Ananya Rao', 'patient@healthplain.ai', '+91 9876543210', 'direct', 'patient', 1, ?)",
                (datetime.utcnow().isoformat(),)
            )

        if conn.execute("SELECT COUNT(*) AS c FROM doctors").fetchone()["c"] == 0:
            conn.executemany(
                "INSERT INTO doctors (name, specialty, experience, verified, photo_url) VALUES (?,?,?,?,?)",
                [
                    ("Dr. Sarah Jenkins, MD", "Cardiologist", 12, 1, "https://images.unsplash.com/photo-1537368910025-700350fe46c7?auto=format&fit=crop&w=500&q=80"),
                    ("Dr. Karthik Varma, MBBS", "Internal Medicine", 9, 1, "https://images.unsplash.com/photo-1622253692010-333f2da6031d?auto=format&fit=crop&w=500&q=80"),
                    ("Dr. Meera Nair, MD", "Endocrinologist", 11, 0, "https://images.unsplash.com/photo-1594824476967-48c8b964273f?auto=format&fit=crop&w=500&q=80"),
                ]
            )

        if conn.execute("SELECT COUNT(*) AS c FROM patients_directory").fetchone()["c"] == 0:
            conn.executemany(
                "INSERT INTO patients_directory (name, age, phone, condition_summary, created_at) VALUES (?,?,?,?,?)",
                [
                    ("Ananya Rao", 34, "+91 98765 43210", "Routine Blood Panel & Vitamin D Review", datetime.utcnow().strftime("%Y-%m-%d")),
                    ("Rohan Deshmukh", 48, "+91 98230 11223", "Hypertension & Lipid Profile Monitoring", datetime.utcnow().strftime("%Y-%m-%d")),
                ]
            )

        if conn.execute("SELECT COUNT(*) AS c FROM appointment_slots").fetchone()["c"] == 0:
            conn.executemany(
                "INSERT INTO appointment_slots (doctor_id, doctor_name, specialty, slot_date, slot_time, status, patient_name, reason) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (1, "Dr. Sarah Jenkins, MD", "Cardiologist", "2026-11-15", "10:00 AM", "Booked", "Ananya Rao", "Cardiology & Blood Report Follow-up"),
                    (1, "Dr. Sarah Jenkins, MD", "Cardiologist", "2026-11-15", "11:30 AM", "Available", None, None),
                    (2, "Dr. Karthik Varma, MBBS", "Internal Medicine", "2026-11-16", "02:30 PM", "Available", None, None),
                ]
            )

        if conn.execute("SELECT COUNT(*) AS c FROM reminders").fetchone()["c"] == 0:
            conn.executemany(
                "INSERT INTO reminders (medicine_name, dosage, schedule_time, status, created_at) VALUES (?,?,?,?,?)",
                [
                    ("Vitamin D3", "1000 IU", "08:30 AM", "Taken", datetime.utcnow().strftime("%Y-%m-%d")),
                    ("Metformin", "500 mg", "08:00 PM", "Pending", datetime.utcnow().strftime("%Y-%m-%d")),
                ]
            )

        if conn.execute("SELECT COUNT(*) AS c FROM recovery_logs").fetchone()["c"] == 0:
            conn.execute(
                "INSERT INTO recovery_logs (patient_name, wellbeing, pain_level, notes, created_at) VALUES (?,?,?,?,?)",
                ("Ananya Rao", "Good", 2, "Energy levels improved after hydration and morning walk.", datetime.utcnow().strftime("%Y-%m-%d %H:%M"))
            )

        if conn.execute("SELECT COUNT(*) AS c FROM prescriptions").fetchone()["c"] == 0:
            conn.execute(
                "INSERT INTO prescriptions (patient_name, medicine_name, dosage, instructions, created_at) VALUES (?,?,?,?,?)",
                ("Ananya Rao", "Vitamin D3 1000 IU", "1 Tablet Daily", "Take after breakfast for 30 days.", datetime.utcnow().strftime("%Y-%m-%d"))
            )


init_db()


def role_required(role_name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            session["role"] = role_name
            session.setdefault("user_id", 1)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(path, ext):
    ext = ext.lower()
    try:
        if ext == "txt":
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        if ext == "docx":
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        if ext in {"png", "jpg", "jpeg"}:
            return pytesseract.image_to_string(Image.open(path))
        if ext == "pdf":
            chunks = []
            with fitz.open(path) as pdf:
                for page in pdf:
                    text = page.get_text("text").strip()
                    if len(text) < 25:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text = pytesseract.image_to_string(img)
                    chunks.append(text)
            return "\n".join(chunks)
    except Exception as e:
        print(f"Extraction error: {e}")
        return ""
    return ""


def validate_medical_document(text, category, confidence):
    if not text or len(text.strip()) < 20:
        return False, "Document Unclear: Please upload a clearer medical document."

    medical_keywords = [
        "hemoglobin", "glucose", "cholesterol", "platelet", "pathology", "prescription",
        "creatinine", "wbc", "rbc", "diagnosis", "dosage", "tablet", "capsule", "clinic",
        "hospital", "laboratory", "specimen", "serum", "plasma", "mg/dl", "g/dl", "test"
    ]
    has_medical_terms = any(kw in text.lower() for kw in medical_keywords)

    reject_keywords = ["invoice", "shopping", "resume", "curriculum vitae", "syllabus", "lecture notes"]
    has_reject_terms = any(rk in text.lower() for rk in reject_keywords)

    if has_reject_terms and not has_medical_terms:
        return False, "Invalid Document: This file does not appear to be a valid medical report."

    if (category != "medical" or confidence < 0.65) and not has_medical_terms:
        return False, "Invalid Document: Please upload a valid medical report or prescription."

    return True, "Validated Medical Document"


def classify_document(text):
    if not MODEL_PATH.exists():
        return "medical", 0.92
    model = joblib.load(MODEL_PATH)
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < 20:
        return "unreadable", 0.0
    probs = model.predict_proba([cleaned])[0]
    best = int(probs.argmax())
    return str(model.classes_[best]), float(probs[best])


TEST_PATTERNS = {
    "Hemoglobin": r"\b(?:haemoglobin|hemoglobin|hb)\b[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)\s*(g/dl|g/dL)?",
    "Blood glucose": r"\b(?:glucose|blood sugar|fasting sugar)\b[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)\s*(mg/dl|mg/dL)?",
    "Total cholesterol": r"\b(?:total cholesterol|cholesterol)\b[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)\s*(mg/dl|mg/dL)?",
    "Creatinine": r"\bcreatinine\b[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)\s*(mg/dl|mg/dL)?",
    "Platelets": r"\bplatelets?\b[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)\s*(?:10\^?3/?µ?l|thousand/?µ?l)?",
    "WBC": r"\b(?:wbc|white blood cells?)\b[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)\s*(?:10\^?3/?µ?l|thousand/?µ?l)?",
}


def extract_lab_values(text):
    found = []
    for label, pattern in TEST_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            val = match.group(1)
            unit = match.group(2) if (match.lastindex and match.lastindex >= 2 and match.group(2)) else "standard unit"
            found.append({"test": label, "value": val, "unit": unit})
    return found


@app.route("/")
def index():
    session.pop("role", None)
    return render_template("index.html")


@app.route("/enter/<role>")
def enter_role(role):
    if role not in ("patient", "doctor", "admin"):
        return redirect(url_for("index"))
    session["role"] = role
    session["user_id"] = 1
    if role == "patient":
        return redirect(url_for("patient_dashboard"))
    elif role == "doctor":
        return redirect(url_for("doctor_dashboard"))
    else:
        return redirect(url_for("admin_dashboard"))


@app.route("/set-language/<lang>")
def set_language(lang):
    if lang in ("en", "te"):
        session["lang"] = lang
    return redirect(request.referrer or url_for("index"))


# --- PATIENT ROUTES (6 TABS) ---

@app.route("/patient/dashboard")
@role_required("patient")
def patient_dashboard():
    with db() as conn:
        reports = conn.execute("SELECT * FROM reports ORDER BY id DESC LIMIT 5").fetchall()
        reminders = conn.execute("SELECT * FROM reminders ORDER BY id DESC").fetchall()
        appointments = conn.execute("SELECT * FROM appointment_slots WHERE status='Booked' ORDER BY id DESC").fetchall()
        checkins = conn.execute("SELECT * FROM recovery_logs ORDER BY id DESC LIMIT 3").fetchall()
    return render_template("dashboard.html", active_tab="dashboard", reports=reports, reminders=reminders, appointments=appointments, checkins=checkins)


@app.route("/patient/simplifier")
@role_required("patient")
def patient_simplifier():
    with db() as conn:
        reports = conn.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
    return render_template("patient_simplifier.html", active_tab="simplifier", reports=reports)


@app.route("/upload", methods=["POST"])
@role_required("patient")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        flash("Please select a medical document to upload.")
        return redirect(url_for("patient_simplifier"))

    if not allowed_file(file.filename):
        flash("Supported formats: PDF, DOCX, TXT, PNG, JPG.")
        return redirect(url_for("patient_simplifier"))

    original_name = secure_filename(file.filename) or f"report_{uuid.uuid4().hex[:8]}"
    ext = original_name.rsplit(".", 1)[1].lower()
    stored_name = f"{uuid.uuid4().hex}.{ext}"
    path = UPLOAD_DIR / stored_name
    file.save(path)

    try:
        text = extract_text(path, ext)
        category, confidence = classify_document(text)
        is_valid, message = validate_medical_document(text, category, confidence)

        if not is_valid:
            if path.exists():
                path.unlink()
            flash(message)
            return redirect(url_for("patient_simplifier"))

        values = extract_lab_values(text)
        status = "Simplified & Ready for Review"

        with db() as conn:
            conn.execute(
                """INSERT INTO reports
                   (user_id, original_name, stored_name, file_type, predicted_category, confidence,
                    extracted_text, extracted_values, status, created_at)
                   VALUES (1,?,?,?,?,?,?,?,?,?)""",
                (original_name, stored_name, ext, category, confidence,
                 text[:10000], json.dumps(values), status, datetime.utcnow().isoformat())
            )
        flash("Medical document simplified successfully.")
    except Exception as exc:
        if path.exists():
            path.unlink()
        flash(f"Processing error: {exc}")

    return redirect(url_for("patient_simplifier"))


@app.route("/report/<int:report_id>")
def report_detail(report_id):
    with db() as conn:
        report = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    if not report:
        flash("Report not found.")
        return redirect(url_for("index"))
    values = json.loads(report["extracted_values"] or "[]")
    return render_template("report.html", report=report, values=values)


@app.route("/patient/reminders", methods=["GET", "POST"])
@role_required("patient")
def patient_reminders():
    with db() as conn:
        if request.method == "POST":
            action = request.form.get("action", "add")
            if action == "toggle":
                rem_id = request.form.get("reminder_id")
                conn.execute("UPDATE reminders SET status = CASE WHEN status='Taken' THEN 'Pending' ELSE 'Taken' END WHERE id=?", (rem_id,))
                flash("Reminder status updated.")
            else:
                med = request.form.get("medicine_name", "").strip()
                dosage = request.form.get("dosage", "").strip()
                time_str = request.form.get("schedule_time", "").strip()
                if med and dosage and time_str:
                    conn.execute(
                        "INSERT INTO reminders (medicine_name, dosage, schedule_time, status, created_at) VALUES (?,?,?,'Pending',?)",
                        (med, dosage, time_str, datetime.utcnow().strftime("%Y-%m-%d"))
                    )
                    flash("Medicine reminder added.")
            return redirect(url_for("patient_reminders"))

        reminders = conn.execute("SELECT * FROM reminders ORDER BY id DESC").fetchall()
    return render_template("patient_reminders.html", active_tab="reminders", reminders=reminders)


@app.route("/patient/wellness")
@role_required("patient")
def patient_wellness():
    return render_template("patient_wellness.html", active_tab="wellness")


@app.route("/patient/find-doctors", methods=["GET", "POST"])
@role_required("patient")
def find_doctors():
    with db() as conn:
        if request.method == "POST":
            slot_id = request.form.get("slot_id")
            patient_name = request.form.get("patient_name", "Ananya Rao").strip() or "Ananya Rao"
            reason = request.form.get("reason", "General Consultation").strip() or "General Consultation"
            conn.execute(
                "UPDATE appointment_slots SET status='Booked', patient_name=?, reason=? WHERE id=?",
                (patient_name, reason, slot_id)
            )
            flash("Appointment booked successfully.")
            return redirect(url_for("find_doctors"))

        doctors = conn.execute("SELECT * FROM doctors WHERE verified=1 ORDER BY id ASC").fetchall()
        slots = conn.execute("SELECT * FROM appointment_slots ORDER BY slot_date ASC, slot_time ASC").fetchall()
    return render_template("patient_find_doctors.html", active_tab="find_doctors", doctors=doctors, slots=slots)


@app.route("/patient/recovery", methods=["GET", "POST"])
@role_required("patient")
def recovery_checkin():
    with db() as conn:
        if request.method == "POST":
            wellbeing = request.form.get("wellbeing", "Good")
            pain_level = int(request.form.get("pain_level", 2))
            notes = request.form.get("notes", "").strip()
            conn.execute(
                "INSERT INTO recovery_logs (patient_name, wellbeing, pain_level, notes, created_at) VALUES ('Ananya Rao',?,?,?,?)",
                (wellbeing, pain_level, notes, datetime.utcnow().strftime("%Y-%m-%d %H:%M"))
            )
            flash("Daily recovery check-in recorded.")
            return redirect(url_for("recovery_checkin"))

        logs = conn.execute("SELECT * FROM recovery_logs ORDER BY id DESC").fetchall()
    return render_template("patient_recovery.html", active_tab="recovery", logs=logs)


# --- DOCTOR ROUTES (4 TABS) ---

@app.route("/doctor/dashboard")
@role_required("doctor")
def doctor_dashboard():
    with db() as conn:
        reports = conn.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
        appointments = conn.execute("SELECT * FROM appointment_slots WHERE status='Booked' ORDER BY id DESC").fetchall()
        checkins = conn.execute("SELECT * FROM recovery_logs ORDER BY id DESC").fetchall()
        prescriptions = conn.execute("SELECT * FROM prescriptions ORDER BY id DESC").fetchall()
    return render_template(
        "dashboard.html",
        active_tab="dashboard",
        reports=reports,
        appointments=appointments,
        checkins=checkins,
        prescriptions=prescriptions
    )


@app.route("/doctor/appointments", methods=["GET", "POST"])
@role_required("doctor")
def doctor_appointments():
    with db() as conn:
        if request.method == "POST":
            slot_id = request.form.get("slot_id")
            new_status = request.form.get("status", "Completed")
            conn.execute("UPDATE appointment_slots SET status=? WHERE id=?", (new_status, slot_id))
            flash(f"Appointment marked as {new_status}.")
            return redirect(url_for("doctor_appointments"))

        appointments = conn.execute("SELECT * FROM appointment_slots ORDER BY slot_date ASC").fetchall()
    return render_template("doctor_appointments.html", active_tab="appointments", appointments=appointments)


@app.route("/doctor/prescriptions", methods=["GET", "POST"])
@role_required("doctor")
def doctor_prescriptions():
    with db() as conn:
        if request.method == "POST":
            patient_name = request.form.get("patient_name", "").strip()
            medicine_name = request.form.get("medicine_name", "").strip()
            dosage = request.form.get("dosage", "").strip()
            instructions = request.form.get("instructions", "").strip()
            if patient_name and medicine_name and dosage:
                conn.execute(
                    "INSERT INTO prescriptions (patient_name, medicine_name, dosage, instructions, created_at) VALUES (?,?,?,?,?)",
                    (patient_name, medicine_name, dosage, instructions, datetime.utcnow().strftime("%Y-%m-%d"))
                )
                conn.execute(
                    "INSERT INTO reminders (medicine_name, dosage, schedule_time, status, created_at) VALUES (?,?,?,?,?)",
                    (medicine_name, dosage, "09:00 AM", "Pending", datetime.utcnow().strftime("%Y-%m-%d"))
                )
                flash("Prescription issued and synced to patient reminders.")
            return redirect(url_for("doctor_prescriptions"))

        prescriptions = conn.execute("SELECT * FROM prescriptions ORDER BY id DESC").fetchall()
        reports = conn.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
    return render_template("doctor_prescriptions.html", active_tab="prescription", prescriptions=prescriptions, reports=reports)


@app.route("/add-doctor-note/<int:report_id>", methods=["POST"])
@role_required("doctor")
def add_doctor_note(report_id):
    notes = request.form.get("notes", "").strip()
    with db() as conn:
        conn.execute("UPDATE reports SET doctor_notes=?, status='Verified by Doctor' WHERE id=?", (notes, report_id))
    flash("Clinical notes saved.")
    return redirect(url_for("report_detail", report_id=report_id))


@app.route("/doctor/recovery")
@role_required("doctor")
def doctor_recovery():
    with db() as conn:
        logs = conn.execute("SELECT * FROM recovery_logs ORDER BY id DESC").fetchall()
    return render_template("doctor_recovery.html", active_tab="recovery", logs=logs)


# --- HOSPITAL ADMIN ROUTES (4 TABS) ---

@app.route("/admin/dashboard")
@role_required("admin")
def admin_dashboard():
    with db() as conn:
        total_doctors = conn.execute("SELECT COUNT(*) AS c FROM doctors WHERE verified=1").fetchone()["c"]
        total_patients = conn.execute("SELECT COUNT(*) AS c FROM patients_directory").fetchone()["c"]
        open_slots = conn.execute("SELECT COUNT(*) AS c FROM appointment_slots WHERE status='Available'").fetchone()["c"]
        total_reports = conn.execute("SELECT COUNT(*) AS c FROM reports").fetchone()["c"]
        doctors = conn.execute("SELECT * FROM doctors ORDER BY id DESC").fetchall()
        slots = conn.execute("SELECT * FROM appointment_slots ORDER BY id DESC LIMIT 5").fetchall()
    return render_template(
        "dashboard.html",
        active_tab="dashboard",
        total_doctors=total_doctors,
        total_patients=total_patients,
        open_slots=open_slots,
        total_reports=total_reports,
        doctors=doctors,
        slots=slots
    )


@app.route("/admin/doctors", methods=["GET", "POST"])
@role_required("admin")
def admin_doctors():
    with db() as conn:
        if request.method == "POST":
            action = request.form.get("action", "add")
            if action == "verify":
                doc_id = request.form.get("doctor_id")
                conn.execute("UPDATE doctors SET verified=1 WHERE id=?", (doc_id,))
                flash("Doctor verified successfully.")
            else:
                name = request.form.get("name", "").strip()
                specialty = request.form.get("specialty", "").strip()
                exp = int(request.form.get("experience", 5))
                if name and specialty:
                    conn.execute(
                        "INSERT INTO doctors (name, specialty, experience, verified, photo_url) VALUES (?,?,?,1,?)",
                        (name, specialty, exp, "https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?auto=format&fit=crop&w=500&q=80")
                    )
                    flash("Doctor added and verified.")
            return redirect(url_for("admin_doctors"))

        doctors = conn.execute("SELECT * FROM doctors ORDER BY id DESC").fetchall()
    return render_template("admin_doctors.html", active_tab="doctors", doctors=doctors)


@app.route("/admin/patients", methods=["GET", "POST"])
@role_required("admin")
def admin_patients():
    with db() as conn:
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            age = int(request.form.get("age", 30))
            phone = request.form.get("phone", "").strip()
            condition = request.form.get("condition_summary", "").strip()
            if name:
                conn.execute(
                    "INSERT INTO patients_directory (name, age, phone, condition_summary, created_at) VALUES (?,?,?,?,?)",
                    (name, age, phone, condition, datetime.utcnow().strftime("%Y-%m-%d"))
                )
                flash("Patient record added.")
            return redirect(url_for("admin_patients"))

        patients = conn.execute("SELECT * FROM patients_directory ORDER BY id DESC").fetchall()
    return render_template("admin_patients.html", active_tab="patients", patients=patients)


@app.route("/admin/slots", methods=["GET", "POST"])
@role_required("admin")
def admin_slots():
    with db() as conn:
        if request.method == "POST":
            doctor_id = int(request.form.get("doctor_id"))
            slot_date = request.form.get("slot_date", "").strip()
            slot_time = request.form.get("slot_time", "").strip()
            doc = conn.execute("SELECT * FROM doctors WHERE id=?", (doctor_id,)).fetchone()
            if doc and slot_date and slot_time:
                conn.execute(
                    "INSERT INTO appointment_slots (doctor_id, doctor_name, specialty, slot_date, slot_time, status) VALUES (?,?,?,?,?,'Available')",
                    (doc["id"], doc["name"], doc["specialty"], slot_date, slot_time)
                )
                flash("Appointment slot published.")
            return redirect(url_for("admin_slots"))

        doctors = conn.execute("SELECT * FROM doctors WHERE verified=1 ORDER BY name ASC").fetchall()
        slots = conn.execute("SELECT * FROM appointment_slots ORDER BY slot_date ASC, slot_time ASC").fetchall()
    return render_template("admin_slots.html", active_tab="slots", doctors=doctors, slots=slots)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
