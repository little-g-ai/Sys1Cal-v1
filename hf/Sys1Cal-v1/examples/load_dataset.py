from datasets import load_dataset


# Replace "RiccardoPorcedda/Sys1Cal-v1" with the final repo id if needed.
parallel = load_dataset(
    "RiccardoPorcedda/Sys1Cal-v1",
    "parallel_primitives",
    split="test",
)

binary = load_dataset(
    "RiccardoPorcedda/Sys1Cal-v1",
    "binary",
    split="test",
)

print(parallel[0])
print(binary[0])
