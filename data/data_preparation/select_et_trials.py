import os
import pandas as pd
from data.data_configs import (
    ALL_SUBJECTS,
    LABELED_ET_ALL_DATA_DIR,
    SELECTED_ET_DATA_DIR,
)


def extract_high_quality_et_trials(subjects=ALL_SUBJECTS, min_trials=250):
    fixation_summary = pd.read_csv(
        LABELED_ET_ALL_DATA_DIR / "fixation_summary_all.csv",
        dtype={"stimulus": str}, 
    )

    stats_rows = []
    valid_subjects = []
    invalid_subjects = {}
    selected_summary_dfs = {}  # subject -> selected fixation-summary df

    for subject in subjects:

        if subject not in fixation_summary["subject"].values:
            invalid_subjects[subject] = "no_data_available"
            continue

        subject_summary = fixation_summary[
            fixation_summary["subject"] == subject
        ].copy()

        n_non_plane_trials = (subject_summary["n_airplanes"] == 0).sum()
        if n_non_plane_trials not in (48, 49):
            invalid_subjects[subject] = f"unexpected_non_plane_trials_{n_non_plane_trials}"
            continue

        if subject_summary.shape[0] != 501:
            invalid_subjects[subject] = f"unexpected_trial_count_{subject_summary.shape[0]}"
            continue

        valid_trial_mask = (
            (subject_summary["n_airplanes"] == 0) |
            (
                (subject_summary["n_airplanes"] == subject_summary["n_fixated_airplanes"]) &
                (subject_summary["n_fixations_on_airplane"] > 0) &
                (subject_summary["n_fixations_not_on_airplane"] > 0)
            )
        )

        valid_summary_trials = subject_summary[valid_trial_mask].copy()
        if valid_summary_trials.shape[0] < min_trials:
            invalid_subjects[subject] = f"too_few_valid_trials_{valid_summary_trials.shape[0]}"
            continue

        non_plane_trials = valid_summary_trials[
            valid_summary_trials["n_airplanes"] == 0
        ]

        plane_trials = valid_summary_trials[
            valid_summary_trials["n_airplanes"] != 0
        ].copy()

        plane_trials["abs_diff"] = (
            plane_trials["n_fixations_on_airplane"]
            - plane_trials["n_fixations_not_on_airplane"]
        ).abs()

        n_to_keep = max(min_trials - len(non_plane_trials), 0)

        selected_plane_trials = (
            plane_trials
            .sort_values("abs_diff", ascending=False)
            .head(n_to_keep)
            .drop(columns="abs_diff")
        )

        selected_summary = pd.concat(
            [non_plane_trials, selected_plane_trials]
        ).reset_index(drop=True)

        # ensure stimulus stays string
        selected_summary["stimulus"] = selected_summary["stimulus"].astype(str)

        fix_on = selected_summary["n_fixations_on_airplane"].sum()
        fix_off = selected_summary["n_fixations_not_on_airplane"].sum()
        total_fix = fix_on + fix_off

        stats_rows.append({
            "subject": subject,
            "n_trials_final": len(selected_summary),
            "fix_on_airplane": fix_on,
            "fix_not_on_airplane": fix_off,
            "prop_on_airplane": (fix_on / total_fix * 100) if total_fix > 0 else 0,
            "prop_not_on_airplane": (fix_off / total_fix * 100) if total_fix > 0 else 0,
        })

        valid_subjects.append(subject)
        selected_summary_dfs[subject] = selected_summary

    stats_df = pd.DataFrame(stats_rows)

    return stats_df, valid_subjects, invalid_subjects, selected_summary_dfs


def select_et_trials(subjects=ALL_SUBJECTS):

    os.makedirs(SELECTED_ET_DATA_DIR, exist_ok=True)

    (
        stats_df,
        valid_subjects,
        invalid_subjects,
        selected_summary_dfs,
    ) = extract_high_quality_et_trials(subjects)

    # ------------------
    # Save global statistics
    # ------------------
    stats_df.to_csv(
        SELECTED_ET_DATA_DIR / "selected_et_statistics.csv",
        index=False,
    )

    # ------------------
    # Save valid/invalid subjects
    # ------------------
    with open(SELECTED_ET_DATA_DIR / "valid_subjects.txt", "w") as f:
        f.write("\n".join(valid_subjects))

    with open(SELECTED_ET_DATA_DIR / "invalid_subjects.txt", "w") as f:
        for subject, reason in invalid_subjects.items():
            f.write(f"{subject}: {reason}\n")

    # ------------------
    # Save per-subject selected event-level data
    # ------------------
    for subject, selected_summary in selected_summary_dfs.items():

        labeled_events = pd.read_csv(
            LABELED_ET_ALL_DATA_DIR / f"{subject}_labeled_events.csv",
            dtype={"stimulus": str},  # ensure stimulus keeps leading zeros
        )

        selected_mask = labeled_events["stimulus"].isin(
            selected_summary["stimulus"]
        )

        selected_events = labeled_events[selected_mask].copy()
        selected_events["stimulus"] = selected_events["stimulus"].astype(str)

        selected_events.to_csv(
            SELECTED_ET_DATA_DIR / f"{subject}_labeled_events.csv",
            index=False,
        )

    # ------------------
    # Save filtered fixation summary for selected subjects/trials
    # ------------------
    fixation_summary_all = pd.read_csv(
        LABELED_ET_ALL_DATA_DIR / "fixation_summary_all.csv",
        dtype={"stimulus": str},
    )
    filtered_fixation_summary = []
    for subject, selected_summary in selected_summary_dfs.items():
        selected_stimuli = set(selected_summary["stimulus"])
        filtered = fixation_summary_all[
            (fixation_summary_all["subject"] == subject) &
            (fixation_summary_all["stimulus"].astype(str).isin(selected_stimuli))
        ]
        filtered_fixation_summary.append(filtered)
        
    if filtered_fixation_summary:
        filtered_df = pd.concat(filtered_fixation_summary, ignore_index=True)
        filtered_df.to_csv(
            SELECTED_ET_DATA_DIR / "fixation_summary_all.csv",
            index=False,
        )

