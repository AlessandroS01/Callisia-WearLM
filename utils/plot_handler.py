import altair as alt
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.dalia.ecg.ecg_cleaning_type import ECGProcessMethod


class PlotHandler:

    def __init__(self, path_data: str):
        """
            Constructor for the PlotHandler class

            Args:
                path_data: Path to the dataset
        """

        self.dataset = pd.read_csv(path_data)


    def create_ecg_plot(self, start_row=0, window_size=4000, method: str = 'neurokit'):
        """
            Generates ECG plot of the given window

            Args:
                start_row: Start row of the ECG plot
                window_size: Window size of the ECG plot
                method: Method of the ECG plot. Default is 'neurokit'
        """

        end_row = start_row + window_size

        ecg_subset = np.array(self.dataset[start_row:end_row]).squeeze()

        pd.DataFrame(ecg_subset).plot()

        for ecg_method in ECGProcessMethod:
            signals, info = nk.ecg_process(ecg_subset, sampling_rate=700, method=ecg_method.value)
            nk.ecg_plot(signals, info)
            print(info)

            pd.DataFrame({f"ECG_NeuroKit_{ecg_method.value}": signals['ECG_Clean']}).plot()

        #pd.DataFrame(ecg_subset).plot()

        #for method in ECGCleaningMethod:
        #    pd.DataFrame({
        #        f"ECG_NeuroKit_{method.value}": nk.ecg_clean(ecg_subset, sampling_rate=700, method=method.value),
        #    }).plot()

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