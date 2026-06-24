from pathlib import Path
import subprocess

import os


os.chdir("/Users/yingyan_zhao/Dropbox/JournalPublicationProject")

OUTPUT_DIR = Path(
    os.environ.get(
        "REPEC_OUTPUT_DIR",
        "/Users/yingyan_zhao/Dropbox/JournalPublicationProject/data/raw/repec/RePEc-ReDIF",
    )
)
REPEC_RSYNC_URL = "rsync://rsync.repec.org/RePEc-ReDIF/"
EXCLUDE_PATTERNS = [
    "*/dpmptsp.bengkuluprov.go.id/***",
    "/imb/Faculty/***",
    "*.pdf",
    "*.PDF",
    "*.zip",
    "*.ZIP",
    "*.doc",
    "*.docx",
    "*.xls",
    "*.xlsx",
    "*.ppt",
    "*.pptx",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        "rsync",
        "-va",
        "--delete",
        *exclude_options(EXCLUDE_PATTERNS),
        REPEC_RSYNC_URL,
        str(OUTPUT_DIR),
    ]

    print("Running:")
    print(" ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        if error.returncode == 23:
            print("\nRePEc download finished with rsync code 23.")
            print("This means some files or attributes were not transferred.")
            print("For RePEc, this can happen because a few remote archives have broken paths or permissions.")
            print("Most downloaded metadata is still usable; you can usually continue to parsing.")
            print(f"Partial RePEc ReDIF metadata is in {OUTPUT_DIR}")
            return

        print("\nRePEc download failed.")
        print("If the error mentions a long path, add that archive path to EXCLUDE_PATTERNS.")
        print("The script already excludes the known broken dpmptsp.bengkuluprov.go.id paths.")
        raise error
    print(f"Downloaded RePEc ReDIF metadata to {OUTPUT_DIR}")


def exclude_options(patterns: list[str]) -> list[str]:
    options = []
    for pattern in patterns:
        options.extend(["--exclude", pattern])
    return options


if __name__ == "__main__":
    main()
