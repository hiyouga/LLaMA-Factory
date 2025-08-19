#!/usr/bin/env python3
"""FSDP Configuration Validator CLI Tool.

Standalone utility to validate FSDP configurations before training.
"""

import argparse
import sys
from pathlib import Path


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llamafactory.hparams.fsdp_validator import FsdpValidator


def main():
    parser = argparse.ArgumentParser(description="Validate FSDP configuration files")
    parser.add_argument("config_file", help="Path to the accelerate configuration file")
    parser.add_argument("--suggest-migration", action="store_true", help="Show suggestions for migrating to FSDP v2")
    parser.add_argument("--quiet", action="store_true", help="Only show errors, suppress success messages")

    args = parser.parse_args()

    try:
        # Validate the configuration
        config = FsdpValidator.validate_config_file(args.config_file)

        if not args.quiet:
            print(f"✅ Configuration file '{args.config_file}' is valid")

        if config.distributed_type == "FSDP" and config.fsdp_config:
            fsdp_version = getattr(config.fsdp_config, "fsdp_version", 1)
            if not args.quiet:
                print(f"📋 FSDP version: {fsdp_version}")
                print(f"📋 Sharding strategy: {getattr(config.fsdp_config, 'fsdp_sharding_strategy', 'N/A')}")
                print(f"📋 Parameter offloading: {getattr(config.fsdp_config, 'fsdp_offload_params', False)}")
                print(f"📋 State dict type: {getattr(config.fsdp_config, 'fsdp_state_dict_type', 'N/A')}")

            # Show migration suggestions if requested or if using v1
            if args.suggest_migration or fsdp_version == 1:
                import yaml

                with open(args.config_file) as f:
                    config_dict = yaml.safe_load(f)

                suggestions = FsdpValidator.suggest_fsdp_v2_migration(config_dict)
                if suggestions:
                    print("\n💡 FSDP v2 Migration Suggestions:")
                    for param, message in suggestions.items():
                        print(f"   • {param}: {message}")

                    if fsdp_version == 1:
                        print("\n🚀 Quick migration command:")
                        print(
                            f"   accelerate to-fsdp2 --config_file {args.config_file} --output_file {args.config_file.replace('.yaml', '_v2.yaml')}"
                        )

        return 0

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}", file=sys.stderr)

        # Provide helpful suggestions based on error type
        error_str = str(e)
        if "does not support these parameters" in error_str:
            print("\n💡 Tip: Use 'accelerate to-fsdp2' to convert FSDP v1 configs to v2", file=sys.stderr)
        elif "Config file not found" in error_str:
            print("\n💡 Tip: Create config with 'accelerate config'", file=sys.stderr)
        elif "Invalid YAML" in error_str:
            print(f"\n💡 Tip: Check YAML syntax in {args.config_file}", file=sys.stderr)

        return 1


if __name__ == "__main__":
    sys.exit(main())
