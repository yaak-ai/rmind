import faiss
import tqdm
import torch
import argparse
import random

from pathlib import Path


def build_index(features, centroids=1024, sub_quantizers=16, bits=8):
    n, d = features.shape

    quantizer = faiss.IndexFlatL2(d)
    print(f"Built quantizer {type(quantizer)}")
    index = faiss.IndexIVFPQ(quantizer, d, centroids, sub_quantizers, bits)
    print(f"Built indexer {type(index)}")
    index.train(features)

    return index


def get_features(filelist, silent=False):
    features = []
    clips = []

    pbar = tqdm.tqdm(filelist, ascii=True) if not silent else filelist
    for filepath in pbar:
        f = torch.load(filepath.strip())
        feat = f["features"]
        n, t, d = feat.shape
        feat = feat.view(n, 1, t * d)
        features.append(feat)
        samples = [
            f"{drive}/frames/cam_front_left.defish.mp4/576x324/{frame[0]:09}.jpg"
            for drive, frame in zip(f["drive_id"], f["frame_idx"])
        ]
        clips.extend(samples)

    features = torch.cat(features)
    n, t, d = features.shape
    features = features.view(n * t, d)

    return features.numpy(), clips


def construct_index(feature_list, limit, frac_split, index_file, metadata_file):
    with feature_list.open("r") as pfile:
        feature_files = pfile.readlines()

    feature_files = feature_files[:limit]
    split = int(len(feature_files) * frac_split)
    train_files = feature_files[:split]
    add_files = feature_files[split:]
    features, _ = get_features(train_files)

    index = build_index(features)

    pbar = tqdm.tqdm(add_files, ascii=True)

    clips = []
    for filepath in pbar:
        f, c = get_features([filepath], silent=True)
        index.add(f)
        clips.extend(c)
        pbar.set_description(f"Adding {filepath.strip()} to the index")

    faiss.write_index(index, f"{index_file}")
    print(f"Wrote index {index_file}")

    assert index.ntotal == len(
        clips
    ), f"Index has {index.ntotal} and clips info {len(clips)}"

    with metadata_file.open("w") as pfile:
        pfile.writelines("\n".join(clips) + "\n")

    qindex = 0
    qv, qc = get_features(random.sample(feature_files, 1))
    D, I = index.search(qv[[qindex]], k=1)

    print(f"Query {qc[qindex]} matches {clips[I[0, 0]]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build a faiss index")
    parser.add_argument(
        "-l", "--list", dest="list", type=Path, help="train feature list"
    )
    parser.add_argument(
        "-x",
        "--limit",
        dest="limit",
        default=100,
        type=int,
        help="limit training of index to X features .pt files",
    )
    parser.add_argument(
        "-s",
        "--split",
        dest="split",
        default=0.2,
        type=float,
        help="fraction of vector used to training",
    )
    parser.add_argument(
        "-i",
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

    args = parser.parse_args()

    construct_index(
        args.list,
        args.limit,
        args.split,
        args.index_file,
        args.metadata_file,
    )
