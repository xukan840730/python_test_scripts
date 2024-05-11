import os
import shutil


def sync_single_file(src_dir_path, file_name, dst_dir_path):
    src_file_path = os.path.join(src_dir_path, file_name)
    assert os.path.isfile(src_file_path)
    dst_file_path = os.path.join(dst_dir_path, file_name)
    if not os.path.exists(dst_file_path) or os.path.getsize(src_file_path) != os.path.getsize(dst_file_path):
        shutil.copyfile(src_file_path, dst_file_path)
        print(f'{src_file_path} -> {dst_file_path}')


def sync_dir_recursively(src_dir_path, dst_dir_path):
    if not os.path.exists(dst_dir_path):
        os.mkdir(dst_dir_path)

    l_dir = os.listdir(src_dir_path)
    for item in l_dir:
        src_file_path = os.path.join(src_dir_path, item)
        if os.path.isfile(src_file_path):
            sync_single_file(src_dir_path, item, dst_dir_path)
        elif os.path.isdir(src_file_path):
            new_dst_dir_path = os.path.join(dst_dir_path, item)
            sync_dir_recursively(src_file_path, new_dst_dir_path)


if __name__ == '__main__':
    sync_dir_recursively('d:\\sync_test_src', 'd:\\sync_test_dst')
