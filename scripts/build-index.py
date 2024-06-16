import argparse
from pathlib import Path

import faiss
import torch
import tqdm


def build_index(features, centroids=1024, sub_quantizers=16, bits=8):
    _n, d = features.shape

    index_name = f"IVF{centroids},PQ{sub_quantizers}x{bits}"
    # https://github.com/facebookresearch/faiss/wiki/The-index-factory
    index = faiss.index_factory(d, index_name)
    index.train(features)

    return index


def get_features(filelist, objective, token, silent=False):
    features = []
    clips = []

    pbar = tqdm.tqdm(filelist, ascii=True) if not silent else filelist
    for filepath in pbar:
        f = torch.load(filepath.strip())
        feat = f["features"][f"{objective}/{token}"]
        n, _, _, d = feat.shape
        feat = feat.view(n, d)
        features.append(feat)
        samples = [
            f"/nas/drives/yaak/data/{drive}/frames/cam_front_left.defish.mp4/576x324/{frame[0]:09}.jpg"
            for drive, frame in zip(f["drive_id"], f["frame_idx"])
        ]
        clips.extend(samples)

    features = torch.cat(features)

    return features.numpy(), clips


def construct_index(
    feature_list,
    limit,
    frac_split,
    index_file,
    metadata_file,
    objective,
    token,
    centroids,
    sub_quantizers,
    bits,
):
    with feature_list.open("r") as pfile:
        feature_files = pfile.readlines()

    pt_files = feature_files[:limit]
    split = int(len(pt_files) * frac_split)
    train_files = pt_files[:split]
    add_files = pt_files[split:]
    features, _ = get_features(train_files, objective, token)

    index = build_index(
        features, centroids=centroids, sub_quantizers=sub_quantizers, bits=bits
    )

    add_files_chunks = [add_files[i : i + 100] for i in range(0, len(add_files), 100)]
    pbar = tqdm.tqdm(add_files_chunks, ascii=True)

    clips = []
    for filepaths in pbar:
        f, c = get_features(filepaths, objective, token, silent=True)
        index.add(f)
        clips.extend(c)
        pbar.set_description(f"Adding {len(filepaths) * 100} features to the index")

    faiss.write_index(index, f"{index_file}")

    assert index.ntotal == len(
        clips
    ), f"Index has {index.ntotal} and clips info {len(clips)}"

    with metadata_file.open("w") as pfile:
        pfile.writelines("\n".join(clips) + "\n")


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
    parser.add_argument(
        "-o",
        "--objective",
        required=True,
        type=str,
        dest="objective",
        help="which objective's feature to use",
    )
    parser.add_argument(
        "-t",
        "--token",
        required=True,
        type=str,
        dest="token",
        help="which token's embeddings to use",
    )
    parser.add_argument(
        "-c",
        "--centroids",
        type=int,
        default=1024,
        dest="centroids",
        help="centroids in each sub-group",
    )
    parser.add_argument(
        "-q",
        "--sub_quantizers",
        type=int,
        default=16,
        dest="sub_quantizers",
        help="Sub quantizers in faiss",
    )
    parser.add_argument(
        "-b", "--bits", type=int, default=8, dest="bits", help="bits for IVF"
    )

    args = parser.parse_args()

    construct_index(
        args.list,
        args.limit,
        args.split,
        args.index_file,
        args.metadata_file,
        args.objective,
        args.token,
        args.centroids,
        args.sub_quantizers,
        args.bits,
    )
