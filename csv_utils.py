"""
Created by Dimitrios Karageorgiou
email: dkarageo@iti.gr
Mar 31, 2024
"""

from pathlib import Path
import logging
import csv
from shutil import copyfile
from typing import Any, Optional

import tqdm


logger = logging.getLogger(__name__)


class AlgorithmOutputData:
    def __init__(self,
                 tampered: Optional[dict[Path, Path]],
                 untampered: Optional[dict[Path, Path]],
                 response_times: Optional[dict[Path, float]],
                 image_level_predictions: Optional[dict[Path, float]] = None):
        # Mapping between source tampered files and output files.
        self.tampered: Optional[dict[Path, Path]] = tampered
        # Mapping between source untampered files and output ones
        self.untampered: Optional[dict[Path, Path]] = untampered
        self.image_level_predictions: Optional[dict[Path, float]] = image_level_predictions
        self.response_times: Optional[dict[Path, float]] = response_times

    def save_masks(self, output_dir: Path) -> 'AlgorithmOutputData':
        """Copy masks to a dir, with their filenames matching the inputs.

        :returns: A new AlgorithmOutputData object with the paths of the masks matching
            the copied files.
        """
        new_manipulated: dict[Path, Path] = {}
        new_authentic: dict[Path, Path] = {}

        if self.tampered:
            tampered_masks_dir: Path = output_dir / "manipulated_masks"
            tampered_masks_dir.mkdir(exist_ok=True)
            for image, mask in tqdm.tqdm(self.tampered.items(),
                                         desc="Copying masks of manipulated images",
                                         unit="image"):
                target: Path = tampered_masks_dir / f"{image.stem}{mask.suffix}"
                copyfile(mask, target)
                new_manipulated[image] = target
        if self.untampered:
            non_tampered_masks_dir: Path = output_dir / "authentic_masks"
            non_tampered_masks_dir.mkdir(exist_ok=True)
            for image, mask in tqdm.tqdm(self.untampered.items(),
                                         desc="Copying masks of authentic images",
                                         unit="image"):
                target: Path = non_tampered_masks_dir / f"{image.stem}{mask.suffix}"
                copyfile(mask, target)
                new_authentic[image] = target

        new_manipulated: Optional[dict[Path, Path]] = (
            new_manipulated if new_manipulated is not None else None)
        new_authentic: Optional[dict[Path, Path]] = (
            new_authentic if new_authentic is not None else None)
        new_response_times: Optional[dict[Path, float]] = (
            self.response_times.copy() if self.response_times is not None else None)
        new_image_level_predictions: Optional[dict[Path, float]] = (
            self.image_level_predictions.copy() if self.image_level_predictions is not None
            else None
        )

        return AlgorithmOutputData(
            new_manipulated,
            new_authentic,
            new_response_times,
            image_level_predictions=new_image_level_predictions
        )

    def save_csv(
        self,
        csv_path: Path,
        root_path: Path,
        output_column: str,
        image_column: str = "image",
        detection_column: str = "detection",
        positive_value: str = "TRUE",
        negative_value: str = "FALSE"
    ) -> None:
        """Saves references of the algorithm's outputs to a CSV file.

        The references are saved to the column specified by `output_column` and they are
        relative to the `root_path`.

        This method updates the CSV file if it already exists, otherwise it creates a new one.
        A newly created CSV file, apart from the `output_column` will also include the
        following columns:
        - `image_column`
        - `detection_column`
        The `image_column` is expected to be present in already existing CSVs. Also, the
        references for all the images included in the algorithm's outputs are expected
        to be present in the CSV.

        If the algorithm's outputs include image-level detection scores, an additional column
        is added to the CSV. Its header consists of the `image_column` value, followed by
        the `_detection` suffix.

        :param csv_path: Path to the CSV file where the references will be saved.
        :param root_path: Path to the directory to which the saved references will be
            relative to.
        :param output_column: Header of the new column that will be added to the CSV for
            writing the references.
        :param image_column: Column that contains references to the input images.
        :param detection_column: Column that contains the ground-truth label for whether an
            image is manipulated or authentic.
        :param positive_value: Value of the `detection_column` for the manipulated images.
        :param negative_value: Value for the `detection_column` for the authentic images.
        """
        scores_column: str = f"{output_column}_detection"

        if csv_path.exists():  # Update the existing CSV file.
            entries: list[dict[str, str]] = read_csv_file(csv_path)
            # TODO: If the image column contains paths spelled differently than the standard way,
            #  e.g. './path/to/image.jpg', instead of 'path/to/image.jpg', the following
            #  hashing by the raw string value will not work. Should be considered at some point,
            #  but, not very important to consider it now.
            entries_by_image: dict[str, dict[str, str]] = {e[image_column]: e for e in entries}

            if self.tampered is not None:
                for image, prediction in self.tampered.items():
                    image_csv_value: str = str(image.relative_to(root_path))
                    entries_by_image[image_csv_value][output_column] = str(
                        prediction.relative_to(root_path))
                    if self.image_level_predictions and image in self.image_level_predictions:
                        entries_by_image[image_csv_value][scores_column] = str(
                            self.image_level_predictions[image])
            if self.untampered is not None:
                for image, prediction in self.untampered.items():
                    image_csv_value: str = str(image.relative_to(root_path))
                    entries_by_image[image_csv_value][output_column] = str(
                        prediction.relative_to(root_path))
                    if self.image_level_predictions and image in self.image_level_predictions:
                        entries_by_image[image_csv_value][scores_column] = str(
                            self.image_level_predictions[image])
        else:  # Create a new CSV file.
            entries: list[dict[str, str]] = []
            if self.tampered is not None:
                for image, prediction in self.tampered.items():
                    e: dict[str, str] = {
                        image_column: str(image.relative_to(root_path)),
                        detection_column: positive_value,
                        output_column: str(prediction.relative_to(root_path))
                    }
                    if self.image_level_predictions and image in self.image_level_predictions:
                        e[scores_column] = str(self.image_level_predictions[image])
                    entries.append(e)
            if self.untampered is not None:
                for image, prediction in self.untampered.items():
                    e: dict[str, str] = {
                        image_column: str(image.relative_to(root_path)),
                        detection_column: negative_value,
                        output_column: prediction
                    }
                    if self.image_level_predictions and image in self.image_level_predictions:
                        e[scores_column] = str(self.image_level_predictions[image])
                    entries.append(e)
        write_csv_file(entries, csv_path)


def load_dataset_from_csv(
    csv_file: Path,
    root_dir: Optional[Path] = None
) -> tuple[list[Path], list[Path], list[Path]]:
    """Loads a dataset from a csv file.

    The csv file must include the following columns:
    - image: Path to an image file.
    - mask: Path to the ground-truth manipulation localization mask. This field can be
        empty for the authentic images.
    - detection: The label denoting whether an image is authentic or manipulated. 'TRUE',
        'True', 'true' values indicate a manipulated image, while 'FALSE', 'False', 'false'
        values indicate an authentic image.

    :param csv_file: Path to the csv file.
    :param root_dir: Path to the directory to which the paths in the csv file are relative to.
        If this argument is set to None, the directory containing the csv file is used as
        the root directory.

    :returns: A tuple containing in the following order:
        - a list with the paths of the manipulated images,
        - a list with the paths of the authentic images,
        - a list with the paths of the ground-truth manipulation localization masks
            corresponding to the manipulated images.
    """
    image_column: str = "image"
    mask_column: str = "mask"
    detection_column: str = "detection"
    positive_labels: list[str] = ["True", "TRUE", "true"]
    negative_labels: list[str] = ["False", "FALSE", "false"]

    authentic_images: list[Path] = []
    manipulated_images: list[Path] = []
    masks: list[Path] = []

    entries: list[dict[str, str]] = read_csv_file(csv_file)

    if root_dir is None:
        root_dir = csv_file.parent

    for e in entries:
        is_manipulated: bool
        if e[detection_column] in positive_labels:
            is_manipulated = True
        elif e[detection_column] in negative_labels:
            is_manipulated = False
        else:
            raise ValueError(
                f"{e[detection_column]} is not a valid value for the detection field")

        if is_manipulated:
            manipulated_images.append(root_dir/e[image_column])
            masks.append(root_dir/e[mask_column])
        else:
            authentic_images.append(root_dir/e[image_column])

    return manipulated_images, authentic_images, masks


def read_csv_file(csv_file: Path, verbose: bool = True) -> list[dict[str, str]]:
    # Read the whole csv file.
    if verbose:
        logger.info(f"READING CSV: {str(csv_file)}")

    entries: list[dict[str, str]] = []
    with csv_file.open() as f:
        reader: csv.DictReader = csv.DictReader(f, delimiter=",")
        if verbose:
            pbar = tqdm.tqdm(reader, desc="Reading CSV entries", unit="entry")
        else:
            pbar = reader
        for row in pbar:
            entries.append(row)

    if verbose:
        logging.info(f"TOTAL ENTRIES: {len(entries)}")

    return entries


def write_csv_file(data: list[dict[str, Any]], output_file: Path) -> None:
    with output_file.open("w") as f:
        writer: csv.DictWriter = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter=",")
        writer.writeheader()
        for r in data:
            writer.writerow(r)
