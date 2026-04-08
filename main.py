from processing.dalia.feature_extractor import FeatureExtractor


def main():
    extractor()
    #processor = DaliaProcessor("datasets/dalia/standardized/S1")
    #print(processor.get_standardized_windows())

def extractor():
    for i in range(1, 16):
        patient = "S" + str(i)
        r_peaks_path = f"datasets/dalia/standardized/{patient}/rpeaks.csv"
        ecg_path = f"datasets/dalia/standardized/{patient}/chest/chest_ECG.csv"
        fe = FeatureExtractor(r_peaks_path=r_peaks_path, ecg_signal_path=ecg_path)
        fe.calculate_bpm(patient)

if __name__ == '__main__':
    main()