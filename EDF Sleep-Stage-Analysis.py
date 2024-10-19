import mne
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

filename =  "C:/Users/kevin/Desktop/RESEARCH LAB/DummyData.edf"
raw = mne.io.read_raw_edf(filename, preload=True)

raw.filter(l_freq=1.0, h_freq= 200.0, fir_design='firwin')

stage_mapping = {
    'Wake': 1,  # Wake Stages
    'Snore': 2,  # Light Sleep (N1)
    'Deep Breaths In and Out': 3,  # N2
    'Left Foot Movement': 2,  # Light Sleep (N1)
    'Eyes closed': 3,  # N2
    'Breast Breathing': 4,  # Deep Sleep (N3)
}

events = [
    [int(annot['onset'] * raw.info['sfreq']), 0, stage_mapping[annot['description']]]
    for annot in raw.annotations if annot['description'] in stage_mapping
]

events_array = np.array(events)

event_id = {'Wake': 1, 'N1': 2, 'N2': 3, 'N3':4}

mne.viz.plot_events(
    events_array,
    sfreq=raw.info['sfreq'],
    first_samp=raw.first_samp,
    event_id=event_id
)

plt.show()