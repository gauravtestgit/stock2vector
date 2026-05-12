"""Stock2Vec training CLI.

Usage:
    python train.py --market us --mode full
    python train.py --market all --mode incremental
    python train.py --market asx --mode finetune --start 2026-01-01 --end 2026-04-01
"""
import argparse
import logging
import sys

from config.settings import settings


def main(args: argparse.Namespace) -> None:
    """Main training entry point."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Stock2Vec Training")
    logger.info("  Environment: %s", settings.env)
    logger.info("  Market:      %s", args.market)
    logger.info("  Mode:        %s", args.mode)
    logger.info("  Start:       %s", args.start)
    logger.info("  End:         %s", args.end)
    logger.info("  Data dir:    %s", settings.data_dir)
    logger.info("  Embed dim:   %s", settings.embed_dim)
    logger.info("  Epochs:      %s", settings.epochs)
    logger.info("  LR:          %s", settings.lr)

    # TODO: Instantiate ServiceContainer, run pipeline, train, validate, backup
    logger.info("Training not yet implemented — stub only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock2Vec model training")
    parser.add_argument(
        "--market",
        choices=["us", "nzx", "asx", "global", "all"],
        required=True,
        help="Market to train: us, nzx, asx, global, or all",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "finetune"],
        required=True,
        help="Training mode: full, incremental, or finetune",
    )
    parser.add_argument(
        "--start",
        default=settings.training_start,
        help="Training start date (default from settings)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Training end date (default: today)",
    )

    main(parser.parse_args())
