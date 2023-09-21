import faiss
import random
import tqdm
import torch
import argparse

from pathlib import Path


def get_features(filelist, index=-1, silent=False):
    features = []
    clips = []

    pbar = tqdm.tqdm(filelist, ascii=True) if not silent else filelist
    for filepath in pbar:
        f = torch.load(filepath.strip())
        features.append(f["features"][:, [index]])
        samples = [
            f"{drive}/frames/cam_front_left.defish.mp4/576x324/{frame[0]:09}.jpg"
            for drive, frame in zip(f["drive_id"], f["frame_idx"])
        ]
        clips.extend(samples)

    features = torch.cat(features)
    n, t, d = features.shape
    features = features.view(n * t, d)

    return features.numpy(), clips


def search_index(feature_list, index_file, metadata_file, index, query):
    with feature_list.open("r") as pfile:
        feature_files = pfile.readlines()

    index = faiss.read_index(f"{index_file}", faiss.IO_FLAG_READ_ONLY)
    print(f"Loaded index {index_file}")
    with metadata_file.open("r") as pfile:
        clips = [p.strip() for p in pfile.readlines()]
    print(f"Loaded clips {metadata_file}")

    assert index.ntotal == len(
        clips
    ), f"Index has {index.ntotal} and clips info {len(clips)}"

    filepaths = random.sample(feature_files, 1)
    qv, qc = get_features(filepaths)
    D, I = index.search(qv[[query]], k=1)

    print(f"Query {qc[query]} matches {clips[I[0, 0]]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build a faiss index")
    parser.add_argument(
        "-l", "--list", dest="list", type=Path, help="train feature list"
    )
    parser.add_argument(
        "-r",
        "--index",
        dest="index",
        default=-1,
        type=int,
        help="Use feature to index",
    )
    parser.add_argument(
        "-f",
        "--index-file",
        required=True,
        type=Path,
        dest="index_file",
        help="Index file",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        required=True,
        type=Path,
        dest="metadata_file",
        help="metadata file",
    )
    parser.add_argument(
        "-q",
        "--query",
        default=0,
        type=int,
        dest="query",
        help="query",
    )

    args = parser.parse_args()

    search_index(args.list, args.index_file, args.metadata_file, args.index, args.query)
