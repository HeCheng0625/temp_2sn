import typing as tp
import os


def find_tar_offsets(tar_list, num_chunks):
    print(tar_list, len(tar_list), num_chunks)
    tar_num = len(tar_list)
    
    if tar_num < num_chunks:
        num_chunks = tar_num
    
    
    tar_chunk = tar_num // num_chunks
    
    ret_tar_list = [tar_list[i * tar_chunk: (i+1) * tar_chunk] for i in range(num_chunks)]
    
    return ret_tar_list, num_chunks