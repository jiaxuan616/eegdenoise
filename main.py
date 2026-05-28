"""EEGdenoiseNet - Legacy entry point (delegates to new pipeline).

For new usage, prefer:
    python scripts/run_train.py --noise-type EOG
"""


from pipeline import run_full_pipeline


def main():
    print("=" * 60)
    print("EEGdenoiseNet")
    print("=" * 60)
    print()
    print("This legacy main.py now delegates to the new modular pipeline.")
    print("You can also use the new CLI directly:")
    print("    python scripts/run_train.py --noise-type EOG")
    print()

    data_dir = input("data directory [default ./data]: ").strip() or "./data"
    result_location = input("result directory [default ./results]: ").strip() or "./results"

    print()
    print("Please select noise type:")
    print("1. EOG")
    print("2. EMG")
    print("3. EOG_EMG")

    while True:
        choice = input("Enter choice [1-3]: ").strip()
        if choice in ["1", "2", "3"]:
            noise_type = ["EOG", "EMG", "EOG_EMG"][int(choice) - 1]
            break
        print("Invalid choice.")

    run_full_pipeline(
        data_dir=data_dir,
        result_location=result_location,
        noise_type=noise_type,
        epochs=30,
        batch_size=25,
        combin_num=10,
        train_per=0.8,
        optimizer_name="Adam",
        train_num="1",
    )


if __name__ == "__main__":
    main()
