from dataloader.dataset import CityscapesGTA5, SYNTHIA
from dataloader.region_dataset import RegionCityscapesGTA5


def get_dataset(name, data_root, datalist, total_itrs=None, imageset='train'):
    """Obtain a specified dataset class.
    Args:
        name (str): the name of datasets, now only support cityscapes.
        data_root (str): the root directory of data.
        datalist (str): the name of initialized datalist for all mode.
        total_itrs (int): the number of total training iterations.
        imageset (str): "train", "val", "active-label", "active-ulabel" 4 different sets.

    """
    valid_sets = ["train", "val", "active-label", "active-ulabel", "custom-set"]
    valid_datasets = ["cityscapes", "GTA5", "SYNTHIA"]
    if imageset not in valid_sets:
        raise ValueError("Invalid imageset.")
    if name not in valid_datasets:
        raise ValueError("Invalid dataset.")
    if name == "cityscapes" or name == "GTA5":
        dataset = CityscapesGTA5(data_root, datalist, imageset)
    if name == "SYNTHIA":
        dataset = SYNTHIA(data_root, datalist, imageset)
    return dataset


def get_active_dataset(args):
    if args.active_mode == 'scan':
        from dataloader.active_dataset import ActiveDataset
        if args.src_dataset == 'SYNTHIA':
            src_label_dataset = SYNTHIA(args.src_data_dir, args.src_datalist, split='active-label')
        else:
            src_label_dataset = CityscapesGTA5(args.src_data_dir, args.src_datalist, split='active-label')
        trg_label_dataset = CityscapesGTA5(args.src_data_dir, None, split='active-label')
        trg_pool_dataset = CityscapesGTA5(args.trg_data_dir, args.trg_datalist, split='active-ulabel')
        dataset = ActiveDataset(args, src_label_dataset, trg_pool_dataset, trg_label_dataset)
    elif args.active_mode == 'region':
        from dataloader.region_active_dataset import RegionActiveDataset
        if args.src_dataset == 'SYNTHIA':
            src_label_dataset = SYNTHIA(args.src_data_dir, args.src_datalist, split='active-label')
        else:
            src_label_dataset = CityscapesGTA5(args.src_data_dir, args.src_datalist, split='active-label')
        trg_label_dataset = RegionCityscapesGTA5(args.src_data_dir, None, split='active-label')
        trg_pool_dataset = RegionCityscapesGTA5(args.trg_data_dir, args.trg_datalist,
                                                split='active-ulabel', return_spx=True)
        dataset = RegionActiveDataset(args, src_label_dataset, trg_pool_dataset, trg_label_dataset)
    return dataset
