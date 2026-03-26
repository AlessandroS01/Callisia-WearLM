import altair as alt
import neurokit2 as nk
import numpy as np
import pandas as pd


class PlotHandler:

    def __init__(self, path_data: str):
        """
            Constructor for the PlotHandler class

            Args:
                path_data: Path to the dataset
        """

        self.dataset = pd.read_csv(path_data)

    def create_ecg_plot(self, start_row=100000, window_size=10000, method: str = 'neurokit'):
        """
            Generates ECG plot of the given window

            Args:
                start_row: Start row of the ECG plot
                window_size: Window size of the ECG plot
                method: Method of the ECG plot. Default is 'neurokit'
        """
        import matplotlib.pyplot as plt

        end_row = start_row + window_size

        ecg_subset = np.array(self.dataset[start_row:end_row]).squeeze()

        pd.DataFrame(ecg_subset).plot()

        signals = pd.DataFrame({
            "ECG_Raw": ecg_subset,
            "ECG_NeuroKit": nk.ecg_clean(ecg_subset, sampling_rate=700, method="neurokit"),
            "ECG_BioSPPy": nk.ecg_clean(ecg_subset, sampling_rate=700, method="biosppy"),
            "ECG_PanTompkins": nk.ecg_clean(ecg_subset, sampling_rate=700, method="pantompkins1985"),
            "ECG_Hamilton": nk.ecg_clean(ecg_subset, sampling_rate=700, method="hamilton2002"),
            "ECG_Elgendi": nk.ecg_clean(ecg_subset, sampling_rate=700, method="elgendi2010"),
            "ECG_EngZeeMod": nk.ecg_clean(ecg_subset, sampling_rate=700, method="engzeemod2012"),
            "ECG_VG": nk.ecg_clean(ecg_subset, sampling_rate=700, method="vg"),
            "ECG_TC": nk.ecg_clean(ecg_subset, sampling_rate=700, method="templateconvolution")
        })
        signals.plot(subplots= True)
        plt.show()
        #signals, info = nk.ecg_process(ecg_subset, sampling_rate=700)

        #nk.ecg_plot(signals, info)
        #plt.show()

    """def create_plot(self):
        
        #Creates the plot on the given dataset
        
        subset = self.dataset.iloc[::175, :].reset_index()
        chart = alt.Chart(subset).mark_line().encode(
            x=alt.X('index:Q', title='Time'),
            y=alt.Y('ECG:Q', title='ECG Signal'),
        ).properties(
            width=1600,
            height=800,
            title="ECG Signal (Downsampled)"
        ).interactive()

        chart.save('ecg_plot.html')

        return chart"""


    def create_plot(self, start_row=25000, window_size=1000):
        # Slice a consecutive chunk (no skipping points with ::)
        end_row = start_row + 1000
        subset = np.array(self.dataset[start_row:end_row]).squeeze()

        chart = alt.Chart(
            pd.DataFrame({'ECG':nk.signal_filter(subset, sampling_rate=64, highcut=31, method="butterworth")}).reset_index()
        ).mark_line(strokeWidth=1.5).encode(
            x=alt.X('index:Q', title='Sample Index'),
            y=alt.Y('ECG:Q', title='Amplitude', scale=alt.Scale(zero=False))
        ).properties(
            width=1200,
            height=400,
            title=f"ECG Detail: Samples {start_row} to {end_row}"
        ).interactive()

        chart.save('ecg_plot.html')

        return chart