import re
from typing import Tuple

version = "1.0.34"


def split_version_string(version_string: str) -> Tuple[int, int]:
    version_numbers = list(map(int, re.split(r"[-.]", version_string)))
    return version_numbers[0], version_numbers[1]


specification = ".".join([str(i) for i in split_version_string(version)[:2]])


def compatible(serialized_version: str) -> bool:
    self_major, self_minor = split_version_string(version)
    other_major, other_minor = split_version_string(serialized_version)

    if self_major >= other_major:
        return True
    elif self_minor >= other_minor:
        return True
    else:
        return False
