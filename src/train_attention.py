from deep_models import AttentionSleepModel
from deep_training import build_deep_arg_parser, run_deep_training


def main():
    parser = build_deep_arg_parser(default_output_dir="results/attention")
    args = parser.parse_args()
    run_deep_training(
        args=args,
        model_factory=lambda in_channels, num_classes: AttentionSleepModel(
            in_channels=in_channels,
            num_classes=num_classes,
        ),
        model_name="Attention-based Model",
    )


if __name__ == "__main__":
    main()
