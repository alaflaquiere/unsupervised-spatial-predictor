import os
import shutil
from argparse import ArgumentParser
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from network.siamese_network import SiameseSMPredictor
from utils.data_utils import get_dataset_subfolders, get_dataloader, load_regular_grid, normalize_array
from analyze.display_embedding import save_embedding, start_display_server
from utils.misc import get_git_hash, append_time, check_directory


def train_epoch(loader, model, loss_fn, optimizer, scheduler):
    """TODO"""
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    model.train()
    epoch_loss = 0.
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = tuple(x.to(device) for x in inputs)
        targets = targets.to(device)
        outputs = model(*inputs)
        loss = loss_fn(outputs, targets)
        epoch_loss += loss.item()
        for param in model.parameters():  # computationally efficient way to do self.optimizer.zero_grad()
            param.grad = None
        loss.backward()
        optimizer.step()
    scheduler.step()
    return epoch_loss / len(loader)


def send_embedding(model, motor_grid, state_grid, model_dir):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        model.eval()
        h_grid = model.get_representation(motor_grid.to(device))
        h_grid = normalize_array(h_grid.cpu().numpy())
        W = model.get_first_weight_vector().detach().cpu().numpy()
        save_embedding(h_grid,
                       state_grid,
                       W,
                       os.path.join(model_dir, "temp_emb.npy"))


def save_model(model, directory, conf):
    """TODO"""
    torch.save(model.state_dict(),
               os.path.join(directory, "model.pth"))
    with open(os.path.join(directory, "config.yml"), "w") as file:
        yaml.dump(conf, file)


def run(conf):
    """TODO"""
    os.makedirs(conf["save_directory"])
    device = "cuda" if (conf["training"]["gpu"] and torch.cuda.is_available()) else "cpu"
    dir_datasets = get_dataset_subfolders(conf["data_directory"])
    for i, dir_dataset in enumerate(dir_datasets):
        for mode in ["hopping_base", "static_base", "dynamic_base"]:
            for r in range(conf["training"]["repeat_training"]):
                # create the output folder
                model_dir = os.path.join(conf["save_directory"],
                                         "exp{:03}".format(i),
                                         mode,
                                         "trial{:03}".format(r))
                os.makedirs(model_dir)
                # start display server
                if conf["training"]["display"]:
                    display_proc = start_display_server(
                        os.path.join(model_dir, "temp_emb.npy"))
                # get dataloader
                dataset_path = os.path.join(dir_dataset,
                                            "data_{}.npz".format(mode))
                loader, dim_m, dim_s = get_dataloader(dataset_path, conf)
                # load the regular grid for evaluation
                motor_grid, state_grid = load_regular_grid(dir_dataset)
                state_grid = normalize_array(state_grid)
                # create the network
                conf["network"]["dim_m"] = dim_m
                conf["network"]["dim_s"] = dim_s
                net = SiameseSMPredictor(**conf["network"]).to(device)
                # loss
                loss_fn = nn.MSELoss(reduction="mean")
                # optimizer
                optimizer = Adam(net.parameters(),
                                 **conf["optimizer"])
                scheduler = lr_scheduler.ExponentialLR(optimizer,
                                                       1e-2**(1/conf["training"]["n_epochs"]))
                msg = "dataset: {}/{}, {}, trial {}/{}".format(i + 1, len(dir_datasets),
                                                               mode,
                                                               r+1, conf["training"]["repeat_training"])
                for _ in tqdm(range(conf["training"]["n_epochs"]), desc=msg):
                    _ = train_epoch(loader, net, loss_fn, optimizer, scheduler)
                    # visualize the embedding
                    if conf["training"]["display"]:
                        send_embedding(net, motor_grid, state_grid, model_dir)
                # save model
                shutil.copyfile(os.path.join(dir_dataset,
                                             "data_regular_grid.npz"),
                                os.path.join(model_dir,
                                             "data_regular_grid.npz"))
                save_model(net, model_dir, conf)
                # kill display server
                if conf["training"]["display"]:
                    display_proc.kill()
                    os.remove(os.path.join(model_dir, "temp_emb.npy"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="path to config file",
                        default="config\\config.yml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    config["git_commit"] = get_git_hash()
    config["data_directory"] = check_directory(config["data_directory"])
    config["save_directory"] = append_time(config["save_directory"])

    run(config)
