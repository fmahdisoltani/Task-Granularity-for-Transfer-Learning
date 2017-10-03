from torch.nn.parallel import DataParallel


def allocate(variable, device_ids, output_device=None, dim=0):
    if device_ids is None:
        return variable
    elif len(device_ids) == 0:
        return DataParallel(variable, None, output_device, dim)
    else:
        return DataParallel(variable, device_ids, output_device, dim)