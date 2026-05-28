"""Command-line entry point for EEGdenoiseNet.

Usage examples
--------------

    # Run everything with defaults (interactive)
    python scripts/run_train.py

    # Specify noise type and data directory
    python scripts/run_train.py --noise-type EOG --data-dir ./data --result-dir ./results

    # Train only a specific model
    python scripts/run_train.py --noise-type EMG --models Novel_CNN

    # Non-interactive mode with custom hyperparameters
    python scripts/run_train.py --noise-type EOG_EMG --epochs 50 --batch-size 32
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
def main():
    parser = argparse.ArgumentParser(
        description="EEGdenoiseNet — benchmark EEG denoising with deep learning and traditional methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- paths ----
    parser.add_argument("--data-dir", default="./data",
                        help="Directory containing EEG_all_epochs.npy, etc. (default: ./data)")
    parser.add_argument("--result-dir", default="./results",
                        help="Directory for output (default: ./results)")

    # ---- core settings ----
    parser.add_argument("--noise-type", choices=["EOG", "EMG", "EOG_EMG"],
                        default=None, help="Noise type to denoise (if not given, interactive prompt)")
    parser.add_argument("--models", nargs="*",
                        default=None,
                        help="Which deep models to run (default: all five).  "
                             "Choices: fcNN Simple_CNN Complex_CNN RNN_lstm Novel_CNN")

    # ---- hyperparameters ----
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--combin-num", type=int, default=10,
                        help="Random combinations per segment (default: 10)")
    parser.add_argument("--train-per", type=float, default=0.8,
                        help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--optimizer", default="Adam",
                        choices=["Adam", "RMSprop", "SGD"])

    # ---- traditional-only or deep-only modes ----
    parser.add_argument("--traditional-only", action="store_true",
                        help="Run only traditional baselines (Filter + EMD)")
    parser.add_argument("--deep-only", action="store_true",
                        help="Run only deep learning models")

    # ---- run identifier ----
    parser.add_argument("--train-num", default="1",
                        help="Run identifier used as subfolder name (default: '1')")

    args = parser.parse_args()

    # ---- interactive fallback if no noise_type given ----
    if args.noise_type is None:
        print("No noise type specified. Please choose:")
        print("1. EOG")
        print("2. EMG")
        print("3. EOG_EMG")
        while True:
            choice = input("Enter choice [1-3]: ").strip()
            if choice in ["1", "2", "3"]:
                args.noise_type = ["EOG", "EMG", "EOG_EMG"][int(choice) - 1]
                break
            print("Invalid choice. Please enter 1, 2, or 3.")

    # ---- determine which models to run ----
    from pipeline import DEEP_MODELS, run_full_pipeline, run_traditional_baselines, run_deep_model
    from pipeline import _get_datanum_fs, _load_data, prepare_data
    
    if args.traditional_only:
        # Traditional baselines only
        print("="*50)
        print("Running traditional baselines only")
        print("="*50)

        datanum, fs = _get_datanum_fs(args.noise_type)
        EEG_all, noise_all = _load_data(args.data_dir, args.noise_type)

        (
            noiseEEG_train, EEG_train,
            noiseEEG_val, EEG_val,
            noiseEEG_test, EEG_test,
            test_std_VALUE,
        ) = prepare_data(
            EEG_all=EEG_all, noise_all=noise_all,
            combin_num=args.combin_num, train_per=args.train_per,
            noise_type=args.noise_type,
        )

        traditional_folder = f"{args.noise_type}_Traditional"
        run_traditional_baselines(
            noiseEEG_test, EEG_test, test_std_VALUE,
            args.noise_type, fs,
            args.result_dir, traditional_folder, args.train_num,
        )
    elif args.deep_only:
        # Deep models only
        models_to_run = args.models if args.models else DEEP_MODELS
        print("="*50)
        print(f"Running deep models only: {models_to_run}")
        print("="*50)

        datanum, fs = _get_datanum_fs(args.noise_type)
        EEG_all, noise_all = _load_data(args.data_dir, args.noise_type)

        (
            noiseEEG_train, EEG_train,
            noiseEEG_val, EEG_val,
            noiseEEG_test, EEG_test,
            test_std_VALUE,
        ) = prepare_data(
            EEG_all=EEG_all, noise_all=noise_all,
            combin_num=args.combin_num, train_per=args.train_per,
            noise_type=args.noise_type,
        )

        for model_name in models_to_run:
            run_deep_model(
                model_name, datanum, fs,
                noiseEEG_train, EEG_train,
                noiseEEG_val, EEG_val,
                noiseEEG_test, EEG_test,
                test_std_VALUE,
                args.optimizer, args.epochs, args.batch_size,
                args.result_dir, args.noise_type,
                train_num=args.train_num,
            )
    else:
        # Full pipeline
        run_full_pipeline(
            data_dir=args.data_dir,
            result_location=args.result_dir,
            noise_type=args.noise_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            combin_num=args.combin_num,
            train_per=args.train_per,
            optimizer_name=args.optimizer,
            train_num=args.train_num,
            deep_models=args.models,
        )


if __name__ == "__main__":
    main()
