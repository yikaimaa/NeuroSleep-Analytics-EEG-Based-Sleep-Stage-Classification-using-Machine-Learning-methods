from deep_models import USleepModel
from deep_training import build_deep_arg_parser, run_deep_training


def main():
    parser = build_deep_arg_parser(default_output_dir="results/usleep")
    args = parser.parse_args()
    run_deep_training(
        args=args,
        model_factory=lambda in_channels, num_classes: USleepModel(
            in_channels=in_channels,
            num_classes=num_classes,
        ),
        model_name="U-Sleep Model",
    )


if __name__ == "__main__":
    main()
