"""Reject a failing, empty or entirely skipped model suite before publication."""

import argparse
import sys
import xml.etree.ElementTree as ET


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    args = parser.parse_args()
    try:
        cases = list(ET.parse(args.report).iter("testcase"))
        failed = any(
            case.find("failure") is not None or case.find("error") is not None for case in cases
        )
        passed = sum(
            case.find("skipped") is None
            and case.find("failure") is None
            and case.find("error") is None
            for case in cases
        )
        if failed or not passed:
            raise ValueError("Model suite must execute passing tests and have no failures/errors")
    except (OSError, ET.ParseError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"Model suite executed {passed} passing tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
