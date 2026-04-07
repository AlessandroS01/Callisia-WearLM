from processing.dalia_processor import DaliaProcessor

def main():
    processor = DaliaProcessor("datasets/dalia/converted/S1")
    print(processor.get_standardized_windows())


if __name__ == '__main__':
    main()