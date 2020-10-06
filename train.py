import os
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
        # if batch_index % 100 == 99:
        #     print("loss: {}".format(epoch_loss/(batch_index + 1)))
        for param in model.parameters():  # computationally efficient way to do self.optimizer.zero_grad()
            param.grad = None
        loss.backward()
        optimizer.step()
    scheduler.step()
    return epoch_loss / len(loader)


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
            print("\nmode: {}".format(mode))
            for r in range(conf["training"]["repeat_training"]):
                # create the output folder
                model_dir = os.path.join(conf["save_directory"],
                                         "exp{:03}".format(i),
                                         mode,
                                         "trial{:03}".format(r))
                os.makedirs(model_dir)
                print("training {}/{} with {} [{}]".format(r + 1,
                                                             conf["training"]["repeat_training"],
                                                             mode,
                                                             dir_dataset))
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
                motor_grid = motor_grid.to(device)
                # create the network
                net = SiameseSMPredictor(dim_m=dim_m,
                                         dim_s=dim_s,
                                         activation="selu",
                                         **conf["network"]).to(device)
                # loss
                loss_fn = nn.MSELoss(reduction="mean")
                # optimizer
                optimizer = Adam(net.parameters(),
                                 **conf["optimizer"])
                scheduler = lr_scheduler.ExponentialLR(optimizer,
                                                       1e-2**(1/conf["training"]["n_epochs"]))
                for e in tqdm(range(conf["training"]["n_epochs"])):
                    loss = train_epoch(loader, net, loss_fn, optimizer, scheduler)
                    # print("epoch: {} - loss: {}".format(e, loss))
                    # visualize the embedding
                    if conf["training"]["display"]:
                        with torch.no_grad():
                            net.eval()
                            h_grid = net.get_representation(motor_grid)
                            h_grid = h_grid.cpu().numpy()
                            h_grid = normalize_array(h_grid)
                            save_embedding(h_grid,
                                           state_grid,
                                           os.path.join(model_dir, "temp_emb.npy"))
                # save model
                save_model(net, model_dir, conf)
                # kill display server
                if conf["training"]["display"]:
                    display_proc.kill()


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
