
import os

def split_file(file_path, chunk_size):
    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        chunk_number = 0
        while chunk_size * chunk_number < file_size:
            chunk_data = f.read(chunk_size)
            output_file = f"{file_path}.part-{chunk_number:03}"
            with open(output_file, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            print(f"Created {output_file}")
            chunk_number += 1
def combine_files(output_file, input_parts):
    with open(output_file, 'wb') as f:
        for part in input_parts:
            with open(part, 'rb') as chunk:
                f.write(chunk.read())
    print(f"Recombined into {output_file}")

# Example usage
parts = ["path","path2"]
combine_files(r"path", parts)

# Example usage
file_to_split = r"path"
chunk_size_mb = 99  # Specify chunk size in MB
split_file(file_to_split, chunk_size_mb * 1024 * 1024)
