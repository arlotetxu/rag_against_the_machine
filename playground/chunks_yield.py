# Source - https://stackoverflow.com/a/70611137
# Posted by Yohann Boniface, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-02, License - CC BY-SA 4.0

test_file = "indexing/indexer.py"


def chunks(file_name, chunk_size = 100):
    with open(file_name) as f:
        data = f.read()
    size = len(data)
    i = 0
    while i < size:
        yield data[i : i + chunk_size]
        i += chunk_size


if __name__ == '__main__':
    split_files = chunks(test_file)

    for chunk in split_files:
        print(chunk)
        print("==" * 35)


