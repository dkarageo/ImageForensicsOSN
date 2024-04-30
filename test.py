import logging
import os
import copy
import pathlib
import shutil
import timeit
from typing import Any, Generator, Optional

import numpy as np
import cv2
import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.scse import SCSEUnet
import csv_utils


class MyDataset(Dataset):
    def __init__(self, test_path='', size=896):
        self.test_path = test_path
        self.size = size
        self.filelist = sorted(os.listdir(self.test_path))
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return len(self.filelist)

    def load_item(self, idx):
        fname1, fname2 = self.test_path + self.filelist[idx], ''

        img = cv2.imread(fname1)[..., ::-1]
        H, W, _ = img.shape
        mask = np.zeros([H, W, 3])

        H, W, _ = img.shape
        img = img.astype('float') / 255.
        mask = mask.astype('float') / 255.
        return self.transform(img), self.tensor(mask[:, :, :1]), fname1.split('/')[-1]

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.name = 'detector'
        self.det_net = SCSEUnet(backbone_arch='senet154', num_channels=3)

    def forward(self, Ii):
        Mo = self.det_net(Ii)
        return Mo


class Model(nn.Module):
    def __init__(self, device: str = "cuda"):
        super(Model, self).__init__()
        self.save_dir = 'weights/'
        self.networks = Detector()
        self.gen = nn.DataParallel(self.networks, device_ids=[device])

    def forward(self, Ii):
        return self.gen(Ii)

    def load(self, path=''):
        self.gen.load_state_dict(
            torch.load(self.save_dir + path + '%s_weights.pth' % self.networks.name)
        )


@click.command(help="Performs a forensics analysis of images using IFOSN.")
@click.option("--input_dir",
              type=click.Path(exists=True, path_type=pathlib.Path),
              default=pathlib.Path("./data/input"),
              help="Directory that contains the images to be analyzed.")
@click.option("--output_dir",
              type=click.Path(file_okay=False, path_type=pathlib.Path),
              default=pathlib.Path("./data/output"),
              help="Directory where the results will be exported.")
@click.option("--temp_dir",
              type=click.Path(file_okay=False, path_type=pathlib.Path),
              default=pathlib.Path("./temp"))
@click.option("--device", type=str, default="cuda")
@click.option("--chunk_size", type=int,
              help="When chunk size is provided the computation is performed in "
                   "multiple iterations, each processing at most the given number of images. "
                   "The use of this parameter allows to reduce the amount of disk required for "
                   "storing the intermediate files that are generated during the "
                   "IFOSN computation.")
@click.option("--csv_root_dir", "-r",
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
              help="Directory to which paths into the provided input CSV file are relative to. "
                   "This option is only meaningful when the `input_dir` option points to a CSV "
                   "file.")
@click.option("--update_input_csv", is_flag=True, default=False,
              help="When a CSV file is provided as input, providing this flag enables you to "
                   "update the input CSV, instead of exporting a new one.")
def cli(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    temp_dir: pathlib.Path,
    device: str,
    chunk_size: Optional[int],
    csv_root_dir: Optional[pathlib.Path],
    update_input_csv: bool
) -> None:
    # Load model.
    model = Model(device=device)
    model.load()
    model.eval()
    model.to(device)

    temp_dir.mkdir(exist_ok=True, parents=True)

    process_images(
        model=model,
        input_source=input_dir,
        output_dir=output_dir,
        temp_dir=temp_dir,
        device=device,
        chunk_size=chunk_size,
        csv_root_dir=csv_root_dir,
        update_input_csv=update_input_csv
    )


def process_images(
    model,
    input_source: pathlib.Path,
    output_dir: pathlib.Path,
    temp_dir: pathlib.Path,
    device: str,
    chunk_size: Optional[int] = None,
    csv_root_dir: Optional[pathlib.Path] = None,
    update_input_csv: bool = False
) -> None:
    if input_source.is_dir():
        images: list[pathlib.Path] = sorted(list(input_source.iterdir()))
    else:
        forged_images: list[pathlib.Path]
        authentic_images: list[pathlib.Path]
        forged_images, authentic_images, _ = csv_utils.load_dataset_from_csv(
            input_source, csv_root_dir
        )
        images: list[pathlib.Path] = sorted(forged_images + authentic_images)
    images = find_images_without_cached_output(images, output_dir)

    if chunk_size is None:
        images_batches: list[list[pathlib.Path]] = [images]
    else:
        images_batches: list[list[pathlib.Path]] = list(split_in_chunks(images, chunk_size))

    torch.cuda.synchronize()
    start_time: float = timeit.default_timer()

    localization_masks: dict[pathlib.Path, pathlib.Path] = {}
    for batch in images_batches:
        batch_localization_masks: dict[pathlib.Path, pathlib.Path] = forensics_test(
            model, batch, output_dir, temp_dir, device, output_relative_to=csv_root_dir
        )
        localization_masks.update(batch_localization_masks)

    torch.cuda.synchronize()
    stop_time: float = timeit.default_timer()
    elapsed_time: float = stop_time - start_time
    print(f"Total time: {elapsed_time} secs")
    print(f"Time per image: {elapsed_time / len(images)}")

    # In case of a CSV source of images, update it if requested.
    if input_source.is_file() and update_input_csv:
        out_data: csv_utils.AlgorithmOutputData = csv_utils.AlgorithmOutputData(
            tampered=localization_masks,
            untampered=None,
            response_times=None
        )
        out_data.save_csv(input_source, csv_root_dir, "ifosn")


def find_images_without_cached_output(
    images: list[pathlib.Path],
    cache_dir: pathlib.Path
) -> list[pathlib.Path]:
    uncached: list[pathlib.Path] = []

    print(f"Searching cached images in {cache_dir}")

    for img in tqdm(images, desc="Searching cache", unit="image"):
        target_image: pathlib.Path = cache_dir / f"{img.stem}.png"  # Path of expected outputs.
        if not target_image.exists():
            uncached.append(img)

    found: int = len(images) - len(uncached)
    if found > 0:
        print(f"Found {found} images from a previous run.")

    return uncached


def forensics_test(
    model,
    images: list[pathlib.Path],
    output_dir: pathlib.Path,
    temp_dir: pathlib.Path,
    device: str,
    output_relative_to: Optional[pathlib.Path] = None
) -> dict[pathlib.Path, pathlib.Path]:
    """Analyzes a set of images using an IFOSN model.

    :param model: The IFOSN model.
    :param images: The images to analyze.
    :param output_dir: The output directory.
    :param temp_dir: The temporary directory.
    :param device: The device to use.
    :param output_relative_to: A path that belongs to the path tree of all the input images.
        When this path is provided, in the output directory the whole path structure
        below this path, will be maintained. It is useful when the input images are not
        included under a single directory, thus several ones may share the same filename.
        So, retaining the path structure of the input image is required in order to not
        overwrite the result of an image, with the result of another.

    :returns: A dict of paths to the forgery localization masks corresponding to the
        input images.
    """

    test_size: int = 896

    # Decompose the input images.
    decomposition_base_dir: pathlib.Path = temp_dir / f"input_decompose_{test_size}"
    decomposition_dirs: set[pathlib.Path] = decompose(
        images, test_size, decomposition_base_dir, output_relative_to
    )

    # Compute the forgery localization masks of the decomposed images.
    predictions_base_dir: pathlib.Path = temp_dir / f"input_decompose_{test_size}_pred"
    predictions_dirs: set[pathlib.Path] = set()
    for decomp_dir in decomposition_dirs:
        test_dataset = MyDataset(test_path=str(decomp_dir)+"/",
                                 size=int(test_size))
        if output_relative_to is None:
            pred_dir: pathlib.Path = predictions_base_dir
        else:
            pred_dir: pathlib.Path = (
                predictions_base_dir / decomp_dir.relative_to(decomposition_base_dir)
            )
        predictions_dirs.add(pred_dir)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
        pred_dir.mkdir(parents=True, exist_ok=True)
        for items in tqdm(test_loader, desc="Computing IFOSN model outputs", unit="image"):
            Ii, Mg = (item.to(device) for item in items[:-1])
            filename = items[-1]
            Mo = model(Ii)
            Mo = Mo * 255.
            Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()
            for i in range(len(Mo)):
                Mo_tmp = Mo[i][..., ::-1]
                cv2.imwrite(str(pred_dir/f"{filename[i][:-4]}.png"), Mo_tmp)

    # Clean-up decomposition directory.
    if decomposition_base_dir.exists():
        shutil.rmtree(decomposition_base_dir)

    # Merge predictions of the decomposed images into the final predictions for each input image.
    forgery_localization_masks: dict[pathlib.Path, pathlib.Path] = merge(
        images, test_size, output_dir, temp_dir, output_relative_to=output_relative_to
    )

    # Clean-up predictions directory of the decomposed images.
    if predictions_base_dir.exists():
        shutil.rmtree(predictions_base_dir)

    # # Perform evaluation.
    # path_gt = 'data/mask/'
    # if os.path.exists(path_gt):
    #     flist = sorted(os.listdir(path_pre))
    #     auc, f1, iou = [], [], []
    #     for file in flist:
    #         pre = cv2.imread(path_pre + file)
    #         gt = cv2.imread(path_gt + file[:-4] + '.png')
    #         H, W, C = pre.shape
    #         Hg, Wg, C = gt.shape
    #         if H != Hg or W != Wg:
    #             gt = cv2.resize(gt, (W, H))
    #             gt[gt > 127] = 255
    #             gt[gt <= 127] = 0
    #         if np.max(gt) != np.min(gt):
    #             auc.append(roc_auc_score(
    #                 (gt.reshape(H * W * C) / 255).astype('int'), pre.reshape(H * W * C) / 255.)
    #             )
    #         pre[pre > 127] = 255
    #         pre[pre <= 127] = 0
    #         a, b = metric(pre / 255, gt / 255)
    #         f1.append(a)
    #         iou.append(b)
    #     print('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f' % (np.mean(auc), np.mean(f1),
    #                                                              np.mean(iou)))
    return forgery_localization_masks


def decompose(
    images: list[pathlib.Path],
    size: int,
    out_path: pathlib.Path,
    output_relative_to: Optional[pathlib.Path] = None
) -> set[pathlib.Path]:
    decomposition_dirs: set[pathlib.Path] = set()

    for img_path in tqdm(images, desc="Decomposing input images", unit="image"):
        if output_relative_to is None:
            image_decomposition_dir: pathlib.Path = out_path
        else:
            image_decomposition_dir: pathlib.Path = (
                out_path / img_path.parent.relative_to(output_relative_to)
            )
        image_decomposition_dir.mkdir(exist_ok=True, parents=True)
        decomposition_dirs.add(image_decomposition_dir)

        # Load image.
        img: np.ndarray = cv2.imread(str(img_path))
        H, W, _ = img.shape

        # Decompose image into smaller ones with dimensions at most (size, size).
        X, Y = H // (size // 2) + 1, W // (size // 2) + 1
        idx: int = 0
        for x in range(X-1):
            if x * size // 2 + size > H:
                break
            for y in range(Y-1):
                if y * size // 2 + size > W:
                    break
                decomposition_out_path: pathlib.Path = (
                    image_decomposition_dir / f"{img_path.stem}_{idx:03d}.png"
                )
                if not decomposition_out_path.exists():
                    img_tmp = img[x * size // 2: x * size // 2 + size,
                                  y * size // 2: y * size // 2 + size,
                                  :]
                    cv2.imwrite(str(decomposition_out_path), img_tmp)
                idx += 1
            decomposition_out_path: pathlib.Path = (
                image_decomposition_dir / f"{img_path.stem}_{idx:03d}.png"
            )
            if not decomposition_out_path.exists():
                img_tmp = img[x * size // 2: x * size // 2 + size, -size:, :]
                cv2.imwrite(str(decomposition_out_path), img_tmp)
            idx += 1
        for y in range(Y - 1):
            if y * size // 2 + size > W:
                break
            decomposition_out_path: pathlib.Path = (
                image_decomposition_dir / f"{img_path.stem}_{idx:03d}.png"
            )
            if not decomposition_out_path.exists():
                img_tmp = img[-size:, y * size // 2: y * size // 2 + size, :]
                cv2.imwrite(str(decomposition_out_path), img_tmp)
            idx += 1
        decomposition_out_path: pathlib.Path = (
            image_decomposition_dir / f"{img_path.stem}_{idx:03d}.png"
        )
        if not decomposition_out_path.exists():
            img_tmp = img[-size:, -size:, :]
            cv2.imwrite(str(decomposition_out_path), img_tmp)
        idx += 1

    return decomposition_dirs


def merge(
    images: list[pathlib.Path],
    size: int,
    output_dir: pathlib.Path,
    temp_dir: pathlib.Path,
    output_relative_to: Optional[pathlib.Path] = None
) -> dict[pathlib.Path, pathlib.Path]:
    forgery_localization_masks: dict[pathlib.Path, pathlib.Path] = {}

    gk = gkern(size)
    gk = 1 - gk

    for img_path in tqdm(sorted(images), desc="Merging output images", unit="image"):
        if output_relative_to is None:
            image_output_dir: pathlib.Path = output_dir
            path_d: pathlib.Path = temp_dir / f'input_decompose_{size}_pred'
        else:
            image_output_dir: pathlib.Path = (
                output_dir / img_path.parent.relative_to(output_relative_to)
            )
            path_d: pathlib.Path = (
                temp_dir
                / f'input_decompose_{size}_pred'
                / img_path.parent.relative_to(output_relative_to)
            )
        image_output_dir.mkdir(exist_ok=True, parents=True)

        try:
            img = cv2.imread(str(img_path))
            H, W, _ = img.shape
            X, Y = H // (size // 2) + 1, W // (size // 2) + 1
            idx = 0
            rtn = np.ones((H, W, 3), dtype=np.float32) * -1
            for x in range(X-1):
                if x * size // 2 + size > H:
                    break
                for y in range(Y-1):
                    if y * size // 2 + size > W:
                        break
                    img_tmp = cv2.imread(str(path_d/f"{img_path.stem}_{idx:03d}.png"))
                    weight_cur = copy.deepcopy(rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :])
                    h1, w1, _ = weight_cur.shape
                    gk_tmp = cv2.resize(gk, (w1, h1))
                    weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                    weight_cur[weight_cur == -1] = 0
                    weight_tmp = copy.deepcopy(weight_cur)
                    weight_tmp = 1 - weight_tmp
                    rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :] = weight_cur * rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :] + weight_tmp * img_tmp
                    idx += 1
                img_tmp = cv2.imread(str(path_d/f"{img_path.stem}_{idx:03d}.png"))
                weight_cur = copy.deepcopy(rtn[x * size // 2: x * size // 2 + size, -size:, :])
                h1, w1, _ = weight_cur.shape
                gk_tmp = cv2.resize(gk, (w1, h1))
                weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                weight_cur[weight_cur == -1] = 0
                weight_tmp = copy.deepcopy(weight_cur)
                weight_tmp = 1 - weight_tmp
                rtn[x * size // 2: x * size // 2 + size, -size:, :] = weight_cur * rtn[x * size // 2: x * size // 2 + size, -size:, :] + weight_tmp * img_tmp
                idx += 1
            for y in range(Y - 1):
                if y * size // 2 + size > W:
                    break
                img_tmp = cv2.imread(str(path_d/f"{img_path.stem}_{idx:03d}.png"))
                weight_cur = copy.deepcopy(rtn[-size:, y * size // 2: y * size // 2 + size, :])
                h1, w1, _ = weight_cur.shape
                gk_tmp = cv2.resize(gk, (w1, h1))
                weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                weight_cur[weight_cur == -1] = 0
                weight_tmp = copy.deepcopy(weight_cur)
                weight_tmp = 1 - weight_tmp
                rtn[-size:, y * size // 2: y * size // 2 + size, :] = weight_cur * rtn[-size:, y * size // 2: y * size // 2 + size, :] + weight_tmp * img_tmp
                idx += 1
            img_tmp = cv2.imread(str(path_d/f"{img_path.stem}_{idx:03d}.png"))
            weight_cur = copy.deepcopy(rtn[-size:, -size:, :])
            h1, w1, _ = weight_cur.shape
            gk_tmp = cv2.resize(gk, (w1, h1))
            weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
            weight_cur[weight_cur == -1] = 0
            weight_tmp = copy.deepcopy(weight_cur)
            weight_tmp = 1 - weight_tmp
            rtn[-size:, -size:, :] = weight_cur * rtn[-size:, -size:, :] + weight_tmp * img_tmp
            idx += 1
            # rtn[rtn < 127] = 0
            # rtn[rtn >= 127] = 255
            loc_mask_path: pathlib.Path = image_output_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(loc_mask_path), np.uint8(rtn))
            forgery_localization_masks[img_path] = loc_mask_path
        except Exception as e:
            logging.error(f"Failed to merge the outputs of {img_path}")
            logging.exception(e)

    return forgery_localization_masks


def gkern(kernlen=7, nsig=3):
    """Returns a 2D Gaussian kernel."""
    # x = np.linspace(-nsig, nsig, kernlen+1)
    # kern1d = np.diff(st.norm.cdf(x))
    # kern2d = np.outer(kern1d, kern1d)
    # rtn = kern2d/kern2d.sum()
    # rtn = np.concatenate([rtn[..., None], rtn[..., None], rtn[..., None]], axis=2)
    rtn = [[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]]
    rtn = np.array(rtn, dtype=np.float32)
    rtn = np.concatenate([rtn[..., None], rtn[..., None], rtn[..., None]], axis=2)
    rtn = cv2.resize(rtn, (kernlen, kernlen))
    return rtn


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return f1, iou


def split_in_chunks(lst: list[Any], n: int) -> Generator[list, None, None]:
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    cli()
