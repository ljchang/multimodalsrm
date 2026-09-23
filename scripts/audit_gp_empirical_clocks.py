"""Audit clock metadata and native/TR consistency without changing the loader.

The supplied README and files disagree about brain length. This audit records
that disagreement. The separate movie-clock adapter implements the clarified
working rule of retaining the first 252 volumes and trimming the tail.
Outputs contain metadata and aggregate differences, never signal samples.
"""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/Storage/Projects/multimodalsrm/data/EmotionPictures")
    )
    parser.add_argument("--subjects", default="s001,s002")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        status="finished",
        readme_sha256=hashlib.sha256((args.root / "README.txt").read_bytes()).hexdigest(),
        subjects={},
    )
    for subject in args.subjects.split(","):
        directory = args.root / subject
        info = pd.read_csv(next((directory / "brain").glob("*info.csv")))
        image = nib.load(next((directory / "brain").glob("*timecourse.nii.gz")))
        face = pd.read_csv(directory / "face/face_au_tr.csv")
        native = pd.read_csv(directory / "ratings/emotion_ratings_cf.csv")
        binned = pd.read_csv(directory / "ratings/ratings_emotion_tr.csv")
        assert list(native.columns[1:]) == list(binned.columns[1:])
        np.testing.assert_array_equal(native.time_sec, np.arange(len(native)))
        np.testing.assert_array_equal(face.tr, np.arange(len(face)))
        np.testing.assert_array_equal(binned.tr, np.arange(len(binned)))
        paired = native.iloc[:, 1:].to_numpy().reshape(len(binned), 2, -1).mean(axis=1)
        difference = abs(paired - binned.iloc[:, 1:].to_numpy())
        censored = info.censored.to_numpy()
        retained = info.splices.to_numpy() == 0
        row = dict(
            brain_volumes=int(image.shape[-1]),
            brain_info_rows=len(info),
            face_rows=len(face),
            rating_native_rows=len(native),
            rating_tr_rows=len(binned),
            rating_pair_mean_max_absolute_difference=float(np.nanmax(difference)),
            splice_rows=np.flatnonzero(~retained).tolist(),
            non_splice_rows=int(retained.sum()),
            rows_beyond_face_length=len(info) - len(face),
            nonzero_censor_rows_beyond_face_length=int(np.count_nonzero(censored[len(face) :])),
            censor_nonzero_rows=int(np.count_nonzero(censored)),
            censor_fractional_rows=int(np.count_nonzero((censored > 0) & (censored < 1))),
            retained_nonzero_censor_rows=int(np.count_nonzero(censored[retained])),
            retained_post_splice_contamination_rows=int(
                np.count_nonzero(info.post_splice_contam.to_numpy()[retained])
            ),
            current_loader="Remove splice-flagged rows, mask any nonzero censor value, compress remaining rows to a 2 s grid; post-splice flags are not separately masked",
        )
        files = list((directory / "physio").glob("*watching_only.csv"))
        if files:
            clocks = pd.read_csv(files[0], usecols=["time_sec", "time_stitched"])
            original, stitched = clocks.time_sec.to_numpy(), clocks.time_stitched.to_numpy()
            row["physiology"] = dict(
                samples=len(clocks),
                time_sec_range=[float(original.min()), float(original.max())],
                stitched_range=[float(stitched.min()), float(stitched.max())],
                original_steps_above_10ms=int(np.count_nonzero(np.diff(original) > 0.01)),
                stitched_steps_above_10ms=int(np.count_nonzero(np.diff(stitched) > 0.01)),
                total_removed_pause_seconds=float((original - stitched)[-1]),
                current_clock="time_stitched, averaged into 1.5 s bins at bin centers",
                current_signal="EDA - EDA100C-MRI channel, not the separately supplied phasic EDA series",
            )
            pauses = stitched[np.flatnonzero(np.diff(original) > 0.01)]
            splice_rows = np.flatnonzero(~retained)
            if len(pauses) == len(splice_rows):
                original_errors = 2 * splice_rows - pauses
                compressed_errors = 2 * (splice_rows - np.arange(len(splice_rows))) - pauses
                row["pause_marker_alignment"] = dict(
                    boundaries=len(pauses),
                    original_row_clock_errors_seconds=original_errors.tolist(),
                    compressed_clock_errors_seconds=compressed_errors.tolist(),
                    original_max_absolute_error=float(max(abs(original_errors))),
                    compressed_max_absolute_error=float(max(abs(compressed_errors))),
                    final_remaining_observation_shift_seconds=2 * len(splice_rows),
                )
        result["subjects"][subject] = row
    result["interpretation"] = [
        "README describes 252 brain volumes and splice-boundary markers; supplied images/info contain 260 rows with eight splice flags. File metadata alone do not establish the intended trimming rule; the separate movie-clock adapter implements the clarified first-252-row rule.",
        "Independent physiology pause boundaries align with original brain row numbers. Removing flagged observations must be distinguished from compressing their timestamps; the latter creates cumulative clock drift.",
        "Native 1 Hz ratings reproduce supplied 2 s TR ratings by pair averaging; no accumulating native/TR clock discrepancy is supported by that check.",
        "Physiology time_sec includes pause jumps; time_stitched is the continuous watching clock. Replacing stitched time with time_sec would introduce those gaps.",
        "Fractional censor values occur. Current masking treats every nonzero value as censored, rather than checking equality to one.",
        "Post-splice contamination flags and tonic-versus-phasic EDA warrant separate preprocessing review. No preprocessing is changed by this audit.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
