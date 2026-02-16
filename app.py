from flask import Flask, render_template, request
import os
import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents threading crashes in Flask
import matplotlib.pyplot as plt
from scipy.signal import welch
import time

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

def band_power(psd, freqs, fmin, fmax):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

def process_file(filepath, timescale, idx):
    plt.close('all')
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw.filter(1, 40, verbose=False)

    sfreq = raw.info["sfreq"]
    # Convert to Microvolts (uV) for readable band power and scale
    data = raw.get_data(picks=raw.ch_names[:6]) * 1e6 
    ch_names = raw.ch_names[:6]
    times = np.arange(data.shape[1]) / sfreq

    # --- Time Domain Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    offsets = np.arange(len(ch_names)) * 150
    for i in range(len(ch_names)):
        ax.plot(times, data[i] + offsets[i])
    
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names)
    ax.set_xlim(0, timescale)
    ax.set_title(f"EEG Signal – {os.path.basename(filepath)}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (µV)")

    time_plot = f"time_{idx}.png"
    plt.savefig(os.path.join(STATIC_FOLDER, time_plot), dpi=150)
    plt.close()

    # --- PSD + Band Power ---
    psd, freqs = welch(data.mean(axis=0), sfreq, nperseg=2048)
    bands = {
        "Delta (0.5–4 Hz)": band_power(psd, freqs, 0.5, 4),
        "Theta (4–8 Hz)": band_power(psd, freqs, 4, 8),
        "Alpha (8–13 Hz)": band_power(psd, freqs, 8, 13),
        "Beta (13–30 Hz)": band_power(psd, freqs, 13, 30),
        "Gamma (30–40 Hz)": band_power(psd, freqs, 30, 40),
    }

    plt.figure()
    plt.semilogy(freqs, psd)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    psd_plot = f"psd_{idx}.png"
    plt.savefig(os.path.join(STATIC_FOLDER, psd_plot), dpi=150)
    plt.close()

    return {
        "filename": os.path.basename(filepath),
        "channels": len(raw.ch_names),
        "sfreq": sfreq,
        "duration": round(raw.times[-1], 2),
        "time_plot": time_plot,
        "psd_plot": psd_plot,
        "bands": bands
    }

def process_average(filepaths, timescale):
    plt.close('all')
    all_data = []
    ch_names = []
    sfreq = 0
    
    for fp in filepaths:
        raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
        raw.filter(1, 40, verbose=False)
        all_data.append(raw.get_data(picks=raw.ch_names[:6]) * 1e6)
        ch_names = raw.ch_names[:6]
        sfreq = raw.info['sfreq']

    min_len = min(d.shape[1] for d in all_data)
    avg_data = np.mean([d[:, :min_len] for d in all_data], axis=0)
    times = np.arange(avg_data.shape[1]) / sfreq

    fig, ax = plt.subplots(figsize=(12, 6))
    offsets = np.arange(len(ch_names)) * 150
    for i in range(len(ch_names)):
        ax.plot(times, avg_data[i] + offsets[i])
    
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names)
    ax.set_xlim(0, timescale)
    ax.set_title("Grand Average EEG Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (µV)")
    
    avg_filename = f"average_{int(time.time())}.png"
    plt.savefig(os.path.join(STATIC_FOLDER, avg_filename), dpi=150)
    plt.close()
    return avg_filename

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("eegfiles")
    if not files or files[0].filename == "":
        return "No files selected"

    timescale = 1.0
    results, filepaths = [], []

    for idx, file in enumerate(files):
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        filepaths.append(path)
        results.append(process_file(path, timescale, idx))

    avg_plot = process_average(filepaths, timescale) if len(filepaths) > 1 else None
    return render_template("result.html", results=results, filepaths=filepaths, timescale=timescale, avg_plot=avg_plot)

@app.route("/update", methods=["POST"])
def update():
    timescale = float(request.form.get("timescale", 1.0))
    filepaths = request.form.getlist("filepaths")
    
    results = [process_file(fp, timescale, idx) for idx, fp in enumerate(filepaths)]
    avg_plot = process_average(filepaths, timescale) if len(filepaths) > 1 else None
    
    return render_template("result.html", results=results, filepaths=filepaths, timescale=timescale, avg_plot=avg_plot)

if __name__ == "__main__":
    app.run(debug=True)