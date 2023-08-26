# utils for distributed training
import os
from torch.distributed import init_process_group
import socket


def get_free_port():
    """
    look for a free port in the localhost
    """
    sock = socket.socket()
    sock.bind(('', 0))
    port_number = str(sock.getsockname()[1])
    sock.close()
    return port_number


def setup_ddp(rank, world_size, free_port):
    """
    rank: unique id for the current process
    world_size: total number of processes
    world_size = num_nodes * num_gpus
    kraken and Leviathan: single node multiple gpus: world_size = number of gpus
    """
    # single node with multiple gpus: master address will be the localhost and the port will be any
    # free port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = free_port
    # initialize the process group: needed for process to process communication
    init_process_group(backend="nccl", init_method="env://",
                       world_size=world_size, rank=rank)


if __name__ == '__main__':
    f_port = get_free_port()
    print(f"free port: {f_port}")
