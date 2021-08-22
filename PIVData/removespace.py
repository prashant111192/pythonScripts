from glob import glob
from tqdm import tqdm


def main():
    root_dir = "./"
    files = glob(f"{root_dir}/csv_*.csv")
    # files = glob(f"./csv_*.csv")
    for file_ in tqdm(files):
        with open(file_, "r+") as f:
            txt = f.read().replace(" ", "")
            f.seek(0)
            f.write(txt)
            f.truncate()


if __name__ == "__main__":
    main()
